"""
Wan-Video 训练入口。

本文件是 Wan 系列模型的统一训练入口。`examples/wanvideo/model_training/{full,lora,special}/*.sh`
下的预置脚本最终都会调用：

    accelerate launch examples/wanvideo/model_training/train.py ...

高层数据流
----------
输入：
  - 数据集目录（`--dataset_base_path`）与元数据文件（`--dataset_metadata_path`，通常为 `metadata.csv`）。
  - 元数据至少需要包含：
      * `prompt`：用于 SFT 的文本提示词
      * `video`：视频文件路径（相对于数据集根目录）
    额外字段可通过 `--extra_inputs`（逗号分隔）启用。对 TI2V，传入
    `--extra_inputs input_image` 会使用视频首帧作为条件图像。

过程：
  - `UnifiedDataset` 负责加载与预处理视频（裁剪/缩放、采样帧）。
  - `WanTrainingModule` 封装 `WanVideoPipeline`，按顺序执行 pipeline units。
  - 根据 task 选择损失函数（默认：FlowMatch SFT）。
  - 训练由 `accelerate` 驱动（分布式、梯度累积、保存）。

输出：
  - checkpoint 保存到 `--output_path`，文件名为 `epoch-*.safetensors`。
  - 仅导出“可训练参数”（例如 LoRA 权重）；可通过 `--remove_prefix_in_ckpt` 去除前缀（常用 `pipe.dit.`）。
"""

import torch, os, argparse, accelerate, warnings
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    """`WanVideoPipeline` 的训练封装。

    职责：
      - 从本地路径或远程 model id 加载 Wan Pipeline 组件；
      - 可选地按两阶段（data_process/train）拆分 pipeline units（由 `--task` 后缀控制）；
      - 冻结参数、注入 LoRA、切换 scheduler 到训练模式；
      - 将 dataset sample 适配为 pipeline 输入，并计算 task 对应的 loss。
    """
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        """初始化训练模块（由 CLI 参数直接映射）。

        参数分组说明（建议结合预置脚本与文档阅读）：

        1) 模型加载（决定“加载什么/从哪加载”）
          - `model_paths`：本地模型路径列表（JSON 字符串），例如 `'["a.safetensors", "b.safetensors"]'`。
          - `model_id_with_origin_paths`：远程模型条目（逗号分隔 `model_id:pattern`），例如：
              `"Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"`
          二者可混用，最终都会被 `parse_model_configs(...)` 解析为 `ModelConfig` 列表。

          - `tokenizer_path`：tokenizer 的本地目录（不填则使用默认 UMT5 tokenizer 并自动下载）。
          - `audio_processor_path`：音频处理器的本地目录（S2V 任务需要；不填则使用默认并自动下载）。

        2) 训练对象（决定“更新哪些参数”）
          - `trainable_models`：全量训练哪些模块（逗号分隔，如 `"dit"`、`"text_encoder"`）。为空则冻结所有基座参数。

        3) LoRA（决定“LoRA 挂哪/挂哪些层/是否续训/是否差分训练”）
          - `lora_base_model`：往 pipeline 的哪个子模型上挂 LoRA（属性名，如 `"dit"`、`"vace"`）。
          - `lora_target_modules`：逗号分隔的模块名子串列表（如 `"q,k,v,o,ffn.0,ffn.2"`），用于匹配要注入 LoRA 的层。
          - `lora_rank`：LoRA 的秩（r）。
          - `lora_checkpoint`：LoRA 续训/热启动 checkpoint 路径。
          - `preset_lora_path` + `preset_lora_model`：预置 LoRA（先融合再训练），用于差分 LoRA 等训练范式。

        4) 显存相关
          - `use_gradient_checkpointing`：梯度检查点开关。为避免 OOM，即使传 False 也会被强制开启（见下方 Warning）。
          - `use_gradient_checkpointing_offload`：将梯度检查点卸载到 CPU 内存（更省显存，通常更慢）。
          - `fp8_models` / `offload_models`：逗号分隔“条目名”，用于指定哪些模型以 FP8 存储或磁盘 offload。
            条目名需精确匹配：
              * `model_paths` 中的某个 path，或
              * `model_id_with_origin_paths` 中的完整 `model_id:pattern` 字符串。

        5) 数据输入适配
          - `extra_inputs`：逗号分隔的额外输入名列表（如 `"input_image"`、`"control_video,reference_image"`）。
            这些 key 必须能从 dataset sample 中取到；通常需要在训练命令里用 `--data_file_keys` 指定并加载对应字段。

        6) 设备与任务模式
          - `device`：pipeline 计算设备（通常为 `accelerator.device`；可通过 `--initialize_model_on_cpu` 先在 CPU 初始化）。
          - `task`：训练任务名（`sft`/`direct_distill` 等），并支持两阶段后缀：
              * `*:data_process`：仅做前向并输出可缓存中间张量
              * `*:train`：消费缓存中间张量并训练
          - `max_timestep_boundary` / `min_timestep_boundary`：时间步边界，常用于混合/双阶段模型的分段训练。
        """
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/") if audio_processor_path is None else ModelConfig(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        """将 dataset 中的字段映射到 pipeline 的 shared inputs。

        `--extra_inputs` 为逗号分隔列表。常见 Wan 变体约定：
          - `input_image`：使用已加载视频的首帧；
          - `end_image`：使用已加载视频的末帧；
          - `reference_image`/`vace_reference_image`：数据集提供列表时取第 0 张。

        输入：
          - `data`：单条样本 dict（已完成加载/预处理）。
          - `inputs_shared`：将传入 `pipe(...)` 的共享输入（正/负 prompt 共用）。
        输出：
          - 返回补齐额外条件输入后的 `inputs_shared`。
        """
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        """构造 pipeline 的 (shared, positive, negative) 输入三元组。

        输入：
          - `data["video"]`：`UnifiedDataset` 产出的帧列表（list[PIL.Image]）。
          - `data["prompt"]`：提示词字符串。
        输出：
          - `(inputs_shared, inputs_posi, inputs_nega)`：供 unit runner 与 loss 使用。
        """
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        """训练 / 数据预处理任务的 forward。

        - 训练任务（`sft`、`direct_distill` 等）：返回标量 loss tensor。
        - 预处理任务（`*:data_process`）：返回可缓存的中间结果（供拆分训练使用）。
        """
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    """Wan 训练 CLI 参数解析器。

    该 parser 由 `diffsynth.diffusion.parsers` 组合通用训练参数：
      - 数据集 / 模型加载 / 优化器 / 输出 / LoRA / 梯度等配置
    并追加 Wan 专有参数（tokenizer / audio processor / timestep boundary）。
    """
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    # `accelerate` 负责分布式启动、梯度累积与 state_dict 汇聚。
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    # 构建通用数据集：
    # - 从 metadata（csv/json/jsonl）读取每条样本；
    # - 将 `data_file_keys` 指定的字段（通常是图片/视频/音频路径）加载为实际数据对象；
    # - 对视频/图像做统一的裁剪、缩放与采样，以满足模型的尺寸/帧数约束。
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,  # 数据集根目录，用于将 metadata 中的相对路径转为绝对路径
        metadata_path=args.dataset_metadata_path,  # 元数据路径；若为 None 则进入“缓存模式”（递归读取 .pth）
        repeat=args.dataset_repeat,  # 每个 epoch 中将数据集重复的次数（扩大训练步数的简单手段）
        data_file_keys=args.data_file_keys.split(","),  # 需要“按算子加载”的字段名列表；metadata 中不存在的 key 会被忽略
        main_data_operator=UnifiedDataset.default_video_operator(
            # 通用视频/图像加载与预处理算子：
            # - 输入可以是图片/gif/视频路径（字符串）；输出统一为“帧列表”（list[PIL.Image]）
            # - 会根据 height/width/max_pixels 做裁剪缩放；并按 num_frames 对视频前缀采样
            base_path=args.dataset_base_path,  # 与上同：用于 ToAbsolutePath 拼接根目录
            max_pixels=args.max_pixels,  # 动态分辨率时的像素上限：超过则缩小，不足则保持
            height=args.height,  # 固定高度；留空则启用动态分辨率
            width=args.width,  # 固定宽度；留空则启用动态分辨率
            height_division_factor=16,  # 高度对齐到 16 的倍数（与 VAE/模型结构约束有关）
            width_division_factor=16,  # 宽度对齐到 16 的倍数
            num_frames=args.num_frames,  # 训练/推理使用的帧数（从视频前缀采样）
            time_division_factor=4,  # 时间维对齐约束：Wan 常用 4n+1 帧数（例如 49/81/121）
            time_division_remainder=1,
        ),
        special_operator_map={
            # 部分 Wan 变体需要特殊输入，并且其预处理规则固定。
            # 注意：只有当该 key 同时出现在 `data_file_keys` 中时，special operator 才会被触发。
            "animate_face_video": (
                # Animate 任务的人脸视频：固定裁剪/缩放为 512x512（与输出视频分辨率可不同）
                ToAbsolutePath(args.dataset_base_path)
                >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16))
            ),
            "input_audio": (
                # S2V 任务输入音频：读取为波形（默认采样率 16kHz）
                ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000)
            ),
        }
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
