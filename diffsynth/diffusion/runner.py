"""
训练运行器（runner）。

本模块负责“训练循环”以及可选的“数据预处理/缓存”循环。模型相关逻辑（pipeline、loss、
LoRA 注入等）位于 `diffsynth.diffusion.training_module` 与各模型的
`examples/*/model_training/train.py`。

两种常见工作流：
  1) 直接训练（默认）：dataset -> model -> loss -> backward -> optimizer step
  2) 拆分训练：
       - `*:data_process` 阶段将中间张量缓存到磁盘（`.pth`）
       - `*:train` 阶段读取缓存，只训练需要反向传播的部分
"""

import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    """运行标准训练循环。

    输入：
      - `dataset`：每次产出单条样本 dict（通过 `collate_fn` 强制 batch size=1）。
      - `model`：`DiffusionTrainingModule` 子类，forward 返回标量 loss tensor。
      - `accelerator`：负责 DDP、梯度累积与 state_dict 汇聚。
    过程：
      - AdamW 优化器 + 常量 LR scheduler。
      - 每步：forward -> backward -> optimizer.step ->（可选）保存。
    输出：
      - checkpoint 由 `model_logger` 写入其 `output_path`。
    """
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    # 框架默认以 batch size=1 训练；collate_fn 直接取第一个样本。
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    # 拆分训练：`data` 已是缓存的 tuple；model 通过 `inputs=...` 接收。
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    """运行拆分训练的预处理/缓存阶段。

    输入：
      - `dataset`：原始样本（video、prompt、control inputs 等）。
      - `model`：应只运行 forward 的前半段，并返回可缓存的中间张量。
    输出：
      - 每个进程（rank）会写入缓存 `.pth` 文件到：
          `{output_path}/{process_index}/{data_id}.pth`
    """
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
