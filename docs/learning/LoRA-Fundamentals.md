# LoRA 微调技术详解

**LoRA 的解决方案**：
> 冻结原始权重，只训练一个**低秩分解**的小型旁路网络

---

## 1. LoRA 的数学原理

对于原始的线性层：
```
y = W·x    (W 是 d×k 的矩阵)
```

LoRA 将权重更新 ΔW 分解为两个低秩矩阵的乘积：

```
y = W·x + ΔW·x
  = W·x + (B·A)·x

其中：
  W: 原始权重 (d × k)，冻结不训练
  A: 降维矩阵 (r × k)，可训练
  B: 升维矩阵 (d × r)，可训练
  r: LoRA 秩，远小于 d 和 k（如 r=32）
```

**参数量对比**：
```
原始参数量: d × k
LoRA 参数量: r × k + d × r = r × (d + k)

例如 d=4096, k=4096, r=32:
  原始: 4096 × 4096 = 16,777,216 (16M)
  LoRA: 32 × (4096 + 4096) = 262,144 (0.26M)

压缩比: 64 倍！
```


> 图示说明

```
                    ┌─────────────────────────────────────┐
                    │         原始线性层 (冻结)            │
     输入 x ───────►│           W (d × k)                 │───────► 输出 y₁
                    │         requires_grad=False         │
                    └─────────────────────────────────────┘
                                      +
                    ┌─────────────────────────────────────┐
                    │         LoRA 旁路 (可训练)           │
     输入 x ───────►│  A (r × k)  ──►  B (d × r)          │───────► 输出 y₂
                    │         requires_grad=True          │
                    └─────────────────────────────────────┘

                    最终输出 y = y₁ + α/r × y₂

                    α (lora_alpha): 缩放系数，控制 LoRA 的影响强度
```

---

## 2. 实际运行示例

假设我们要微调一个 Attention 层的 Query 投影：

```python
# 原始 Query 投影层
class Attention:
    def __init__(self):
        self.q = nn.Linear(4096, 4096)  # 16M 参数

# 注入 LoRA 后（概念性代码）
class AttentionWithLoRA:
    def __init__(self):
        # 原始权重 - 冻结
        self.q = nn.Linear(4096, 4096)
        self.q.weight.requires_grad = False

        # LoRA 旁路 - 可训练
        self.q_lora_A = nn.Linear(4096, 32, bias=False)  # 降维: 4096 → 32
        self.q_lora_B = nn.Linear(32, 4096, bias=False)  # 升维: 32 → 4096

        # 初始化：A 用高斯，B 用零（确保初始时 LoRA 输出为 0）
        nn.init.kaiming_uniform_(self.q_lora_A.weight)
        nn.init.zeros_(self.q_lora_B.weight)

    def forward(self, x):
        # 原始路径 + LoRA 路径
        return self.q(x) + self.q_lora_B(self.q_lora_A(x))
```

**训练过程**：
```
Step 1: x 通过冻结的 W 得到 y₁
Step 2: x 通过 A → B 得到 y₂
Step 3: y = y₁ + y₂
Step 4: 计算 loss，反向传播
Step 5: 只有 A 和 B 的梯度被计算和更新！
```

---

> 关键参数解释

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `lora_rank` (r) | 低秩分解的秩，越大表达能力越强，但参数越多 | 8, 16, 32, 64 |
| `lora_alpha` (α) | 缩放系数，实际缩放为 α/r | 通常等于 r |
| `target_modules` | 要注入 LoRA 的模块名 | `q,k,v,o,ffn.0,ffn.2` |

---

## 3. DiffSynth-Studio 中的 LoRA 实现

### 3.1 `add_lora_to_model()` - LoRA 注入

位置：`diffsynth/diffusion/training_module.py:56-78`

```python
def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
    # 1. 如果没指定 alpha，默认等于 rank
    if lora_alpha is None:
        lora_alpha = lora_rank

    # 2. 创建 LoRA 配置
    lora_config = LoraConfig(
        r=lora_rank,           # 秩 = 32
        lora_alpha=lora_alpha, # 缩放系数 = 32
        target_modules=target_modules  # ["q", "k", "v", "o", "ffn.0", "ffn.2"]
    )

    # 3. 使用 PEFT 库注入 LoRA（自动找到匹配的层并添加旁路）
    model = inject_adapter_in_model(lora_config, model)

    # 4. 将 LoRA 参数上采样到 bfloat16（训练精度）
    if upcast_dtype is not None:
        for param in model.parameters():
            if param.requires_grad:  # 只有 LoRA 参数是可训练的
                param.data = param.to(upcast_dtype)

    return model
```


### 3.2 `switch_pipe_to_training_mode` 完整流程解析

#### 函数签名与参数

```python
# 位置: diffsynth/diffusion/training_module.py:207-259
def switch_pipe_to_training_mode(
    self,
    pipe,                      # WanVideoPipeline 实例
    trainable_models=None,     # 可训练的模型名（如 "dit"）
    lora_base_model=None,      # 要注入 LoRA 的模型名
    lora_target_modules="",    # LoRA 目标模块
    lora_rank=32,              # LoRA 秩
    lora_checkpoint=None,      # 已有的 LoRA checkpoint（用于继续训练）
    preset_lora_path=None,     # 预置 LoRA（融合到基础模型）
    preset_lora_model=None,    # 预置 LoRA 的目标模型
    task="sft",                # 任务类型
):
```

---

#### Step 1: 设置 Scheduler 为训练模式

```python
pipe.scheduler.set_timesteps(1000, training=True)
```

**作用**：配置 Flow Match Scheduler 的训练时间步

**源码位置**：`diffsynth/diffusion/flow_match.py:132-142`

```python
def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, **kwargs):
    # 生成 sigmas 和 timesteps
    self.sigmas, self.timesteps = self.set_timesteps_fn(
        num_inference_steps=num_inference_steps,  # 训练时 = 1000
        denoising_strength=denoising_strength,
        **kwargs,
    )
    if training:
        self.set_training_weight()  # 设置训练权重
        self.training = True
```

**训练权重计算**（`set_training_weight`）：

```python
def set_training_weight(self):
    steps = 1000
    x = self.timesteps
    # 高斯分布权重：中间时间步权重更高
    y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
    y_shifted = y - y.min()
    bsmntw_weighing = y_shifted * (steps / y_shifted.sum())
    self.linear_timesteps_weights = bsmntw_weighing
```

**图示**：
```
权重
  │     ╭───────╮
  │    ╱         ╲
  │   ╱           ╲
  │  ╱             ╲
  │ ╱               ╲
  └─────────────────────► 时间步
    0    500    1000

中间时间步（~500）权重最高，两端权重较低
这是因为中间时间步的去噪难度最大，需要更多学习
```

---

#### Step 2: 冻结非训练参数

```python
pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
```

**源码位置**：`diffsynth/diffusion/base_pipeline.py:193-202`

```python
def freeze_except(self, model_names):
    # Step 2.1: 将整个 pipeline 设为 eval 模式，冻结所有参数
    self.eval()
    self.requires_grad_(False)

    # Step 2.2: 只解冻指定的模型
    for name in model_names:
        module = self.get_module(self, name)
        if module is None:
            print(f"No {name} models in the pipeline...")
            continue
        module.train()              # 设为训练模式
        module.requires_grad_(True) # 开启梯度计算
```

**对于 LoRA 训练**（`trainable_models=None`）：

```python
pipe.freeze_except([])  # 传入空列表
# 结果：所有模型都被冻结！
# VAE: frozen ✓
# Text Encoder: frozen ✓
# DiT: frozen ✓ (稍后会通过 LoRA 添加可训练参数)
```

**对于全量微调**（`trainable_models="dit"`）：

```python
pipe.freeze_except(["dit"])
# 结果：
# VAE: frozen ✓
# Text Encoder: frozen ✓
# DiT: trainable (所有参数都可训练)
```

---

#### Step 3: 加载预置 LoRA（可选）

```python
if preset_lora_path is not None:
    pipe.load_lora(getattr(pipe, preset_lora_model), preset_lora_path)
```

**用途**：差分 LoRA 训练
- 先加载一个已有的 LoRA 并融合到基础模型
- 然后在此基础上训练新的 LoRA
- 最终效果 = 基础模型 + 预置 LoRA + 新训练的 LoRA
**

**差分 LoRA (Preset LoRA) 详解**

1. 数学表达
    ```
    原始模型:           y = W·x
    加载 Preset LoRA:   y = (W + B₁A₁)·x = W'·x    ← B₁A₁ 被融合到 W 中
    训练新 LoRA:        y = W'·x + B₂A₂·x
                        = (W + B₁A₁)·x + B₂A₂·x
                        = W·x + B₁A₁·x + B₂A₂·x   ← 最终效果
    ```
2. **关键区别**：
   - `B₁A₁` 是**融合**到基础权重 W 中的（变成 W'），不再作为独立旁路存在
   - `B₂A₂` 是新注入的可训练 LoRA
3. 为什么需要差分 LoRA？
   | 场景 | 说明 | 
   |------|------|
   | **风格叠加** | 已有一个"动漫风格" LoRA，想在此基础上训练"特定角色" LoRA |
   | **能力保留** | 已有一个"画质增强" LoRA，想保留画质的同时学习新内容 |
   | **增量学习** | 不想从头训练，而是在已有 LoRA 基础上继续优化 |
   | **LoRA 组合** | 最终可以灵活组合：基础模型 + 风格 LoRA + 角色 LoRA |

```
┌─────────────────────────────────────────────────────────────────┐
│  差分 LoRA 训练流程                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: 加载基础模型                                            │
│  ┌─────────────────┐                                            │
│  │   W (frozen)    │                                            │
│  └─────────────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  Step 2: 融合 Preset LoRA (B₁A₁)                                │
│  ┌─────────────────┐                                            │
│  │ W' = W + B₁A₁   │  ← B₁A₁ 被融合，不再独立存在                │
│  │   (frozen)      │                                            │
│  └─────────────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  Step 3: 注入新 LoRA (B₂A₂)                                     │
│  ┌─────────────────┐     ┌─────────────────┐                    │
│  │ W' (frozen)     │  +  │ B₂A₂ (trainable)│                    │
│  └─────────────────┘     └─────────────────┘                    │
│           │                      │                              │
│           └──────────┬───────────┘                              │
│                      ▼                                          │
│              y = W'·x + B₂A₂·x                                  │
│                                                                 │
│  训练时：只有 B₂ 和 A₂ 的梯度被计算                               │
│  保存时：只保存 B₂A₂（几十 MB）                                   │
│  推理时：可以选择性加载 B₁A₁ 和/或 B₂A₂                          │
└─────────────────────────────────────────────────────────────────┘
```
---

#### Step 4: 注入 LoRA 适配器

```python
if lora_base_model is not None and not task.endswith(":data_process"):
    # 4.1 检查目标模型是否存在
    if (not hasattr(pipe, lora_base_model)) or getattr(pipe, lora_base_model) is None:
        print(f"No {lora_base_model} models in the pipeline...")
        return

    # 4.2 注入 LoRA
    model = self.add_lora_to_model(
        getattr(pipe, lora_base_model),           # pipe.dit
        target_modules=lora_target_modules.split(","),  # ["q","k","v","o","ffn.0","ffn.2"]
        lora_rank=lora_rank,                      # 32
        upcast_dtype=pipe.torch_dtype,            # bfloat16
    )

    # 4.3 如果有 checkpoint，加载已有的 LoRA 权重
    if lora_checkpoint is not None:
        state_dict = load_state_dict(lora_checkpoint)
        state_dict = self.mapping_lora_state_dict(state_dict)
        load_result = model.load_state_dict(state_dict, strict=False)
        print(f"LoRA checkpoint loaded: {lora_checkpoint}")

    # 4.4 将注入 LoRA 后的模型放回 pipeline
    setattr(pipe, lora_base_model, model)
```

**`lora_checkpoint` vs `preset_lora_path` 的区别**


| 参数 | 用途 | 操作方式 | 场景 |
|------|------|----------|------|
| `preset_lora_path` (Step 3) | 预置 LoRA | **融合**到基础模型 | 差分 LoRA |
| `lora_checkpoint` (Step 4) | 断点续训 | **加载**到新 LoRA 适配器 | 继续训练 |

`lora_checkpoint` 的用途：断点续训

```python
# Step 4 中的代码
if lora_checkpoint is not None:
    state_dict = load_state_dict(lora_checkpoint)
    state_dict = self.mapping_lora_state_dict(state_dict)
    load_result = model.load_state_dict(state_dict, strict=False)
```

**场景**：
- 你训练了 3 个 epoch，保存了 `epoch-2.safetensors`
- 想从 epoch 3 继续训练到 epoch 10
- 使用 `--lora_checkpoint ./models/train/xxx/epoch-2.safetensors`

**流程**：
```
1. 注入新的 LoRA 适配器（lora_A, lora_B 初始化为随机/零）
2. 加载 checkpoint 的权重到这些适配器中
3. 继续训练
```

对比图示

```
┌─────────────────────────────────────────────────────────────────┐
│  preset_lora_path (差分 LoRA)                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  基础模型 W ──► 融合 B₁A₁ ──► W' = W + B₁A₁ ──► 注入新 B₂A₂     │
│                                                                 │
│  结果：W' 是新的基础，B₂A₂ 是可训练的                            │
│  用途：在已有 LoRA 基础上训练新能力                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  lora_checkpoint (断点续训)                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  基础模型 W ──► 注入 B₂A₂ ──► 加载 checkpoint 到 B₂A₂           │
│                                                                 │
│  结果：B₂A₂ 从 checkpoint 恢复，继续训练                         │
│  用途：训练中断后继续                                            │
└─────────────────────────────────────────────────────────────────┘
```


---
> 完整流程图

```
switch_pipe_to_training_mode(pipe, lora_base_model="dit", ...)
                │
                ▼
┌───────────────────────────────────────────────────────────────┐
│  Step 1: Scheduler 训练模式                                    │
│  pipe.scheduler.set_timesteps(1000, training=True)            │
│  ├─ 生成 1000 个训练时间步                                      │
│  └─ 计算高斯权重（中间时间步权重更高）                            │
└───────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────┐
│  Step 2: 冻结所有参数                                          │
│  pipe.freeze_except([])                                       │
│  ├─ pipe.eval()                                               │
│  ├─ pipe.requires_grad_(False)                                │
│  │                                                            │
│  │  Pipeline 状态:                                            │
│  │  ┌─────────────────┐                                       │
│  │  │ VAE             │ frozen, requires_grad=False           │
│  │  │ Text Encoder    │ frozen, requires_grad=False           │
│  │  │ DiT             │ frozen, requires_grad=False           │
│  │  └─────────────────┘                                       │
└───────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────┐
│  Step 3: 注入 LoRA                                            │
│  add_lora_to_model(pipe.dit, ["q","k","v","o","ffn.0","ffn.2"])│
│                                                               │
│  DiT 模型变化:                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Transformer Block                                       │  │
│  │  ├─ q: Linear(4096→4096) [frozen]                       │  │
│  │  │   └─ lora_A: Linear(4096→32) [trainable] ← 新增      │  │
│  │  │   └─ lora_B: Linear(32→4096) [trainable] ← 新增      │  │
│  │  ├─ k: Linear(4096→4096) [frozen]                       │  │
│  │  │   └─ lora_A, lora_B [trainable]                      │  │
│  │  ├─ v: Linear(4096→4096) [frozen]                       │  │
│  │  │   └─ lora_A, lora_B [trainable]                      │  │
│  │  ├─ o: Linear(4096→4096) [frozen]                       │  │
│  │  │   └─ lora_A, lora_B [trainable]                      │  │
│  │  ├─ ffn.0: Linear(...) [frozen]                         │  │
│  │  │   └─ lora_A, lora_B [trainable]                      │  │
│  │  └─ ffn.2: Linear(...) [frozen]                         │  │
│  │      └─ lora_A, lora_B [trainable]                      │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────┐
│  最终状态                                                      │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ VAE             │ frozen (不参与训练)                    │  │
│  │ Text Encoder    │ frozen (不参与训练)                    │  │
│  │ DiT             │ 原始权重 frozen + LoRA 参数 trainable  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  可训练参数: 仅 LoRA 的 lora_A 和 lora_B                       │
│  参数量: ~0.1% of 5B = ~5M 参数                               │
└───────────────────────────────────────────────────────────────┘
```
