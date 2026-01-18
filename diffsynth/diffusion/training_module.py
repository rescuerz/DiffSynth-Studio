"""
所有模型训练入口共享的辅助能力。

本模块定义 `DiffusionTrainingModule`：在 `torch.nn.Module` 之上的轻量工具层，提供：
  - 模型配置解析（`--model_paths` / `--model_id_with_origin_paths`）
  - 参数冻结与 LoRA 注入（基于 `peft`）
  - 仅导出可训练参数的 state dict（用于 LoRA/全量微调）
  - 对嵌套输入做递归的 device/dtype 转移

模型训练脚本（例如 `examples/wanvideo/model_training/train.py`）通常需要：
  - 构建 Pipeline（pipe）
  - 调用 `switch_pipe_to_training_mode(...)`
  - 实现 `forward(...)`：执行 pipeline units 并计算任务损失
"""

import torch, json
from ..core import ModelConfig, load_state_dict
from ..utils.controlnet import ControlNetInput
from peft import LoraConfig, inject_adapter_in_model


class DiffusionTrainingModule(torch.nn.Module):
    """扩散模型训练封装的基类。

    本类不负责具体的训练循环；仅提供可复用的“积木”能力（冻结、LoRA 注入、
    state_dict 导出等）。训练循环位于 `diffsynth.diffusion.runner`。
    """
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        """将子模块移动到指定 device/dtype。

        说明：此处刻意遍历 `named_children()`，而不是依赖 `torch.nn.Module.to(...)` 的
        递归行为，因为 Pipeline 往往以属性方式持有模型，且希望显式控制迁移逻辑。
        """
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        """返回 `requires_grad=True` 的参数迭代器。"""
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        """返回可训练参数名集合（`set[str]`）。"""
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        """使用 PEFT 将 LoRA adapter 注入到 `model`。

        输入：
          - `target_modules`：list[str] 或 str；按模块名子串匹配要加 LoRA 的层。
          - `lora_rank`：LoRA 的秩（r）。
          - `lora_alpha`：缩放系数；默认等于 r。
          - `upcast_dtype`：若提供，则将“可训练的 LoRA 参数”上采样到该 dtype。

        输出：
          - 返回注入 LoRA 后的（PEFT 包装）模型，并将 LoRA 参数标记为可训练。
        """
        if lora_alpha is None:
            lora_alpha = lora_rank
        if isinstance(target_modules, list) and len(target_modules) == 1:
            target_modules = target_modules[0]
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        """在不同 PEFT 命名约定之间，对 LoRA state dict 的 key 做归一化。

        旧 checkpoint 可能使用 `lora_A.weight` / `lora_B.weight`；
        新版 PEFT 通常会包含 adapter 名（例如 `lora_A.default.weight`）。
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        """从完整 state_dict 中仅筛出可训练参数。

        `ModelLogger` 会使用该函数保存更小的 checkpoint（例如只保存 LoRA 权重）。
        `remove_prefix` 常设为 `pipe.dit.` 之类的前缀，用于让保存出的 key 更贴近基础模型命名。
        """
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict
    
    
    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        """递归地将嵌套输入移动到 device，并对浮点 tensor 做 dtype 转换。

        支持容器：Tensor / tuple / list / dict。
        非 tensor 对象（例如 PIL Image、字符串）原样返回。
        """
        if data is None:
            return data
        elif isinstance(data, torch.Tensor):
            data = data.to(device)
            if torch_float_dtype is not None and data.dtype in [torch.float, torch.float16, torch.bfloat16]:
                data = data.to(torch_float_dtype)
            return data
        elif isinstance(data, tuple):
            data = tuple(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
            return data
        elif isinstance(data, list):
            data = list(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
            return data
        elif isinstance(data, dict):
            data = {i: self.transfer_data_to_device(data[i], device, torch_float_dtype) for i in data}
            return data
        else:
            return data
    
    def parse_vram_config(self, fp8=False, offload=False, device="cpu"):
        """构造用于 `ModelConfig` 的单模型显存管理配置。

        - `fp8=True`：对“不需要训练的模型参数”使用 FP8 存储（计算仍用 BF16）。
        - `offload=True`：使用磁盘 offload（惰性加载）；要求模型文件为 `.safetensors`。
        """
        if fp8:
            return {
                "offload_dtype": torch.float8_e4m3fn,
                "offload_device": device,
                "onload_dtype": torch.float8_e4m3fn,
                "onload_device": device,
                "preparing_dtype": torch.float8_e4m3fn,
                "preparing_device": device,
                "computation_dtype": torch.bfloat16,
                "computation_device": device,
            }
        elif offload:
            return {
                "offload_dtype": "disk",
                "offload_device": "disk",
                "onload_dtype": "disk",
                "onload_device": "disk",
                "preparing_dtype": torch.bfloat16,
                "preparing_device": device,
                "computation_dtype": torch.bfloat16,
                "computation_device": device,
                "clear_parameters": True,
            }
        else:
            return {}
    
    def parse_model_configs(self, model_paths, model_id_with_origin_paths, fp8_models=None, offload_models=None, device="cpu"):
        """将 CLI 中的模型加载参数解析为 `ModelConfig` 列表。

        输入：
          - `model_paths`：本地路径列表（JSON）。
          - `model_id_with_origin_paths`：以逗号分隔的 `model_id:pattern` 条目。
          - `fp8_models` / `offload_models`：以逗号分隔的条目，匹配对象可以是：
              * `model_paths` JSON 列表中的某个本地 path，或
              * `model_id_with_origin_paths` 中完整的 `model_id:pattern` 字符串
        """
        fp8_models = [] if fp8_models is None else fp8_models.split(",")
        offload_models = [] if offload_models is None else offload_models.split(",")
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            for path in model_paths:
                vram_config = self.parse_vram_config(
                    fp8=path in fp8_models,
                    offload=path in offload_models,
                    device=device
                )
                model_configs.append(ModelConfig(path=path, **vram_config))
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            for model_id_with_origin_path in model_id_with_origin_paths:
                model_id, origin_file_pattern = model_id_with_origin_path.split(":")
                vram_config = self.parse_vram_config(
                    fp8=model_id_with_origin_path in fp8_models,
                    offload=model_id_with_origin_path in offload_models,
                    device=device
                )
                model_configs.append(ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern, **vram_config))
        return model_configs
    
    
    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        task="sft",
    ):
        """将 pipeline 切换到训练模式。

        步骤：
          1) 将 scheduler 切到训练模式（timesteps=1000）。
          2) 冻结除 `trainable_models` 外的所有参数（若提供）。
          3) 可选：加载 preset LoRA（融合到基础模型），用于差分 LoRA 等场景。
          4) 可选：在 `lora_base_model` 上注入可训练 LoRA。

        说明：
          - 对数据预处理任务（`task` 以 `:data_process` 结尾），会跳过 LoRA 注入，
            避免在 worker 上做不必要的模型 patch。
        """
        # Scheduler
        pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Preset LoRA
        if preset_lora_path is not None:
            pipe.load_lora(getattr(pipe, preset_lora_model), preset_lora_path)
        
        # FP8
        # FP8 relies on a model-specific memory management scheme.
        # It is delegated to the subclass.
        
        # Add LoRA to the base models
        if lora_base_model is not None and not task.endswith(":data_process"):
            if (not hasattr(pipe, lora_base_model)) or getattr(pipe, lora_base_model) is None:
                print(f"No {lora_base_model} models in the pipeline. We cannot patch LoRA on the model. If this occurs during the data processing stage, it is normal.")
                return
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(pipe, lora_base_model, model)


    def split_pipeline_units(self, task, pipe, trainable_models=None, lora_base_model=None):
        """为两阶段（data_process/train）工作流拆分 pipeline units。

        - `*:data_process`：保留“需要 backward 的部分”，并缓存中间张量；
        - `*:train`：保留消费缓存张量的“forward/backward 部分”。

        依赖 `pipe.split_pipeline_units(models_require_backward)` 的具体实现。
        """
        models_require_backward = []
        if trainable_models is not None:
            models_require_backward += trainable_models.split(",")
        if lora_base_model is not None:
            models_require_backward += [lora_base_model]
        if task.endswith(":data_process"):
            _, pipe.units = pipe.split_pipeline_units(models_require_backward)
        elif task.endswith(":train"):
            pipe.units, _ = pipe.split_pipeline_units(models_require_backward)
        return pipe
    
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        """解析 ControlNet 风格的条件输入（extra_inputs）。

        约定：
          - 前缀为 `controlnet_` 或 `blockwise_controlnet_` 的 key 会被分组，并转换为
            `ControlNetInput` 对象；
          - 其余 key 会直接拷贝到 `inputs_shared`。
        """
        controlnet_keys_map = (
            ("blockwise_controlnet_", "blockwise_controlnet_inputs",),
            ("controlnet_", "controlnet_inputs"),
        )
        controlnet_inputs = {}
        for extra_input in extra_inputs:
            for prefix, name in controlnet_keys_map:
                if extra_input.startswith(prefix):
                    if name not in controlnet_inputs:
                        controlnet_inputs[name] = {}
                    controlnet_inputs[name][extra_input.replace(prefix, "")] = data[extra_input]
                    break
            else:
                inputs_shared[extra_input] = data[extra_input]
        for name, params in controlnet_inputs.items():
            inputs_shared[name] = [ControlNetInput(**params)]
        return inputs_shared
