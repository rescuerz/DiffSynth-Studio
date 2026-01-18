# 训练循环详解

## 1. `launch_training_task` 函数签名

```python
# 位置: diffsynth/diffusion/runner.py:22-75
def launch_training_task(
    accelerator: Accelerator,           # HuggingFace Accelerate 加速器
    dataset: torch.utils.data.Dataset,  # 数据集
    model: DiffusionTrainingModule,     # 训练模块（包含 pipeline）
    model_logger: ModelLogger,          # 日志/保存器
    learning_rate: float = 1e-5,        # 学习率
    weight_decay: float = 1e-2,         # 权重衰减
    num_workers: int = 1,               # 数据加载线程数
    save_steps: int = None,             # 每 N 步保存（None 则按 epoch 保存）
    num_epochs: int = 1,                # 训练轮数
    args = None,                        # CLI 参数（覆盖上述默认值）
):
```

---

## 2. 训练循环核心代码

```python
# 1. 创建优化器（只优化可训练参数，即 LoRA 参数）
optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

# 2. 创建数据加载器（batch_size=1）
dataloader = torch.utils.data.DataLoader(
    dataset,
    shuffle=True,
    collate_fn=lambda x: x[0],  # 直接取第一个样本
    num_workers=num_workers
)

# 3. 使用 Accelerate 包装（支持分布式训练）
model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

# 4. 训练循环
for epoch_id in range(num_epochs):
    for data in tqdm(dataloader):
        with accelerator.accumulate(model):  # 梯度累积
            optimizer.zero_grad()            # 清零梯度
            loss = model(data)               # 前向传播，计算 loss
            accelerator.backward(loss)       # 反向传播
            optimizer.step()                 # 更新参数
            model_logger.on_step_end(...)    # 可选：按步保存
            scheduler.step()                 # 更新学习率

    if save_steps is None:
        model_logger.on_epoch_end(...)       # 按 epoch 保存
```
> 1. Optimizer（优化器）

优化器负责根据梯度更新模型参数。训练的目标是最小化 loss，优化器决定了"如何根据 loss 的梯度来调整参数"。
```python
optimizer = torch.optim.AdamW(
	model.trainable_modules(),  # 要优化的参数（LoRA 参数）
	lr=learning_rate,           # 学习率
	weight_decay=weight_decay   # 权重衰减
)
```
> 2. Learning Rate（学习率）:控制每次参数更新的步长大小。

最简单的梯度下降：
θ_new = θ_old - lr × gradient

其中：
- θ: 模型参数（如 LoRA 的 A 和 B 矩阵）
- lr: 学习率（如 1e-4 = 0.0001）
- gradient: loss 对参数的梯度（∂Loss/∂θ）

> 3. Weight Decay（权重衰减）

权重衰减是一种正则化方法，用于防止模型过拟合。它通过在损失函数中添加一个正则化项，来惩罚模型参数的绝对值。
```python
loss = loss + weight_decay * torch.norm(model.parameters(), p=2)
```

1. **为什么大参数会导致过拟合？**
   例子：假设我们用多项式拟合数据
   真实规律: y = 2x + 1 (简单线性)
   训练数据（带噪声）:
   ```
		x=1, y=3.1
		x=2, y=5.2
		x=3, y=6.8
	```
	过拟合模型可能学到:
	```
		y = 0.05x³ - 0.3x² + 2.5x + 0.9
	```
	这个模型完美通过所有训练点，但参数很大（0.05, -0.3, 2.5, 0.9）,在新数据上表现会很差
   	**为什么小参数能防止过拟合？**
	```
	大参数 (θ = 100):
	y = 100 × x
	输入 x 的微小变化 → 输出 y 的巨大变化
	x: 1.0 → 1.01  (变化 1%)
	y: 100 → 101   (变化 1，绝对值很大)
	模型对噪声非常敏感，容易过拟合

	小参数 (θ = 1):
	y = 1 × x
	输入 x 的微小变化 → 输出 y 的微小变化
	x: 1.0 → 1.01  (变化 1%)
	y: 1.0 → 1.01  (变化 0.01，绝对值很小)
	模型对噪声不敏感，泛化能力强

	```
1. **Weight Decay 如何解决这个问题？**
   **核心思想：惩罚大参数，让模型倾向于使用小参数。**

	```
	原始 Loss:
	L = MSE(prediction, target)

	加入 Weight Decay 后:
	L_total = MSE(prediction, target) + (λ/2) × ||θ||²
	```
	其中:
	```
	λ = weight_decay (如 0.01)
	||θ||² = Σ θᵢ²  (所有参数的平方和)
	```

	效果：
	- 如果参数 θ 很大，惩罚项 ||θ||² 就很大
	- 优化器会尝试减小 L_total
	- 因此会倾向于让参数变小



> 4. Scheduler（学习率调度器）负责在训练过程中动态调整学习率。

ConstantLR

```python
scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
```
ConstantLR 是最简单的调度器：学习率保持不变。

其他常见调度器

| 调度器            | 行为                 |
|-------------------|----------------------|
| ConstantLR        | 学习率恒定           |
| StepLR            | 每 N 步衰减一次      |
| CosineAnnealingLR | 余弦退火（先快后慢） |
| LinearLR          | 线性衰减             |

> 5. optimizer.step() vs scheduler.step()

optimizer.step() - 更新参数

1. 执行内容：对于每个参数 θ:
   1. 获取梯度 g = θ.grad
   2. 更新动量（Adam 特有）
   3. 计算自适应学习率（Adam 特有）
   4. 更新参数: θ = θ - lr × adjusted_gradient - lr × weight_decay × θ

2. 伪代码：

	```python
	def optimizer.step():
      for param in parameters:
          if param.grad is not None:
              # AdamW 的更新公式（简化版）
              m = beta1 * m + (1 - beta1) * param.grad      # 一阶动量
              v = beta2 * v + (1 - beta2) * param.grad**2   # 二阶动量
              param = param - lr * m / sqrt(v)              # 梯度更新
              param = param - lr * weight_decay * param     # 权重衰减
	```
  
scheduler.step() - 更新学习率

1. 执行内容：根据调度策略更新 optimizer 的学习率

2. 对于 ConstantLR：什么都不做（学习率保持不变）
3. 对于 StepLR：每 N 步将学习率乘以 gamma
4. 对于 CosineAnnealingLR：按余弦曲线调整学习率
---

```
┌─────────────────────────────────────────────────────────────────────────┐
  │                         单步训练流程                                     │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │  Step 1: optimizer.zero_grad()                                         │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │  清零所有参数的梯度                                              │   │
  │  │  for param in parameters:                                       │   │
  │  │      param.grad = 0                                             │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │                              │                                          │
  │                              ▼                                          │
  │  Step 2: loss = model(data)                                            │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │  前向传播：数据 → 模型 → loss                                    │   │
  │  │  PyTorch 自动构建计算图                                          │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │                              │                                          │
  │                              ▼                                          │
  │  Step 3: accelerator.backward(loss)                                    │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │  反向传播：计算 ∂Loss/∂θ                                         │   │
  │  │  for param in parameters:                                       │   │
  │  │      param.grad = ∂Loss/∂param  (链式法则)                      │   │
  │  │                                                                 │   │
  │  │  注意：只有 requires_grad=True 的参数才会计算梯度                │   │
  │  │  即：只有 LoRA 参数有梯度，原始模型参数没有                       │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │                              │                                          │
  │                              ▼                                          │
  │  Step 4: optimizer.step()                                              │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │  更新参数：θ = θ - lr × grad - lr × weight_decay × θ            │   │
  │  │                                                                 │   │
  │  │  LoRA 参数更新示例：                                             │   │
  │  │  lora_A.weight = lora_A.weight - 0.0001 × lora_A.weight.grad   │   │
  │  │  lora_B.weight = lora_B.weight - 0.0001 × lora_B.weight.grad   │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │                              │                                          │
  │                              ▼                                          │
  │  Step 5: scheduler.step()                                              │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │  更新学习率（ConstantLR 不做任何事）                             │   │
  │  │  如果是 CosineAnnealingLR：                                     │   │
  │  │      lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π*t/T))  │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────────────────┘

```

## 3. 完整流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    launch_training_task 训练循环                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Step 1: 初始化                                                  │   │
│  │  ├─ optimizer = AdamW(model.trainable_modules())                │   │
│  │  │   └─ 只优化 LoRA 参数（requires_grad=True 的参数）            │   │
│  │  ├─ scheduler = ConstantLR(optimizer)                           │   │
│  │  └─ dataloader = DataLoader(dataset, batch_size=1)              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Step 2: Accelerate 包装（分布式训练支持）                        │   │
│  │  model, optimizer, dataloader, scheduler = accelerator.prepare() │   │
│  │  ├─ 自动处理 DDP（分布式数据并行）                                │   │
│  │  ├─ 自动处理梯度累积                                             │   │
│  │  └─ 自动处理混合精度训练                                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Step 3: 训练循环                                                │   │
│  │  for epoch in range(num_epochs):                                │   │
│  │      for data in dataloader:                                    │   │
│  │          ┌──────────────────────────────────────────────────┐   │   │
│  │          │  3.1 optimizer.zero_grad()                       │   │   │
│  │          │      └─ 清零 LoRA 参数的梯度                      │   │   │
│  │          ├──────────────────────────────────────────────────┤   │   │
│  │          │  3.2 loss = model(data)                          │   │   │
│  │          │      ├─ 数据预处理（视频 → latents）              │   │   │
│  │          │      ├─ 执行 Pipeline Units                      │   │   │
│  │          │      └─ 计算 FlowMatchSFTLoss                    │   │   │
│  │          ├──────────────────────────────────────────────────┤   │   │
│  │          │  3.3 accelerator.backward(loss)                  │   │   │
│  │          │      └─ 反向传播，计算 LoRA 参数的梯度            │   │   │
│  │          ├──────────────────────────────────────────────────┤   │   │
│  │          │  3.4 optimizer.step()                            │   │   │
│  │          │      └─ 更新 LoRA 参数: θ = θ - lr * grad        │   │   │
│  │          └──────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Step 4: 保存 Checkpoint                                        │   │
│  │  model_logger.on_epoch_end() / on_step_end()                    │   │
│  │  └─ 只保存 LoRA 参数（几十 MB）                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 数据预处理任务 `launch_data_process_task`

除了标准训练循环，还有一个用于拆分训练的预处理任务：

```python
# 位置: diffsynth/diffusion/runner.py:78-108
def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
```

**用途**：将中间张量缓存到磁盘，用于两阶段训练

**流程**：
```
1. 遍历数据集
2. 执行 model.forward() 的前半段（不需要反向传播的部分）
3. 将中间张量保存到 {output_path}/{process_index}/{data_id}.pth
```

**两阶段训练的优势**：
- 第一阶段（data_process）：只做前向计算，缓存中间结果
- 第二阶段（train）：从缓存加载，只训练需要反向传播的部分
- 节省重复计算（如 VAE 编码、Text Encoder 编码）

