# Wan2.2-TI2V-5B LoRA 训练学习路径


## 1. 入口脚本

**文件**: `examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B.sh`

```bash
accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-TI2V-5B_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image"
```

  学习路径建议

  按以下顺序阅读代码：

  | 优先级 | 文件路径                                  | 作用                                                  |
  |--------|-------------------------------------------|-------------------------------------------------------|
  | 1️⃣     | examples/wanvideo/model_training/train.py | 入口文件 - 定义 WanTrainingModule，解析参数，启动训练 |
  | 2️⃣     | diffsynth/diffusion/training_module.py    | 训练基类 - LoRA 注入、模型冻结、训练模式切换          |
  | 3️⃣     | diffsynth/diffusion/runner.py             | 训练循环 - launch_training_task 实现训练 loop         |
  | 4️⃣     | diffsynth/pipelines/wan_video.py          | Pipeline - Wan 模型的推理/训练流水线                  |
  | 5️⃣     | diffsynth/core/data/unified_dataset.py    | 数据集 - 视频数据加载和预处理                         |

  关键执行流程
  ```
    accelerate launch train.py
         │
         ▼
  ┌─────────────────────────────────────┐
  │  1. 解析命令行参数 (wan_parser)      │
  │  2. 创建 Accelerator (分布式训练)    │
  │  3. 创建 UnifiedDataset (加载视频)   │
  │  4. 创建 WanTrainingModule:         │
  │     - 加载预训练模型                 │
  │     - 注入 LoRA 到 DiT 模型          │
  │     - 冻结非训练参数                 │
  │  5. launch_training_task:           │
  │     - 训练循环 (forward → loss → backward) │
  │     - 保存 checkpoint               │
  └─────────────────────────────────────┘

  ---
  ```
  核心概念

  1. LoRA 训练: 只训练 DiT 模型中 q,k,v,o,ffn 层的低秩适配器，参数量小
  2. TI2V (Text+Image to Video): --extra_inputs "input_image" 表示输入包含首帧图像
  3. Flow Match Loss: 使用 FlowMatchSFTLoss 计算训练损失
