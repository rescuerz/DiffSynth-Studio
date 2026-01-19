# Flow Matching SFT Loss 详解

---

## 1. SFT 调用链路

从训练入口到 Loss 计算的完整调用路径：

```
accelerate launch train.py
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  train.py: main                                                         │
│  ├─ WanTrainingModule.__init__()     # 加载模型、注入 LoRA              │
│  │      └─ switch_pipe_to_training_mode()                               │
│  │           └─ scheduler.set_timesteps(1000, training=True)            │
│  │                └─ 初始化 training_weight 权重表                       │
│  └─ launch_training_task()            # 启动训练循环                    │
│         │                                                               │
│         ▼                                                               │
│  runner.py: launch_training_task                                        │
│  └─ for data in dataloader:                                             │
│         loss = model(data)            # 调用 WanTrainingModule.forward  │
│              │                                                          │
│              ▼                                                          │
│  train.py: WanTrainingModule.forward                                    │
│  ├─ get_pipeline_inputs(data)         # 构造 pipeline 输入              │
│  ├─ for unit in pipe.units:           # 执行 pipeline units             │
│  │      inputs = pipe.unit_runner(unit, ...)                            │
│  └─ task_to_loss[task](pipe, *inputs) # 选择并计算 Loss                 │
│              │                                                          │
│              ▼                                                          │
│  loss.py: FlowMatchSFTLoss(pipe, **inputs)                              │
│  └─ 返回标量 loss tensor                                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.1 关键文件定位

| 优先级 | 文件路径 | 作用 |
|--------|----------|------|
| 1️⃣ | `examples/wanvideo/model_training/train.py` | 入口、WanTrainingModule 定义 |
| 2️⃣ | `diffsynth/diffusion/runner.py` | 训练循环 launch_training_task |
| 3️⃣ | `diffsynth/diffusion/loss.py` | FlowMatchSFTLoss 实现 |
| 4️⃣ | `diffsynth/diffusion/flow_match.py` | FlowMatchScheduler 实现 |


---

## 2. Loss 计算调用链路

**FlowMatchSFTLoss 内部调用 scheduler 的方法链：**

```
FlowMatchSFTLoss(pipe, **inputs)
         │
         ├─ 1. torch.randint(...)                    # 采样 timestep 索引
         │      └─ pipe.scheduler.timesteps[id]      # 获取实际 timestep 值
         │
         ├─ 2. torch.randn_like(input_latents)       # 生成随机噪声 ε
         │
         ├─ 3. pipe.scheduler.add_noise(x0, ε, t)    # 构造带噪 latent xt
         │      └─ xt = (1-σ)*x0 + σ*ε
         │
         ├─ 4. pipe.scheduler.training_target(x0, ε, t)  # 构造训练目标
         │      └─ target = ε - x0
         │
         ├─ 5. pipe.model_fn(**inputs, timestep=t)   # DiT 模型预测
         │      └─ noise_pred = DiT(xt, prompt, t)
         │
         ├─ 6. F.mse_loss(noise_pred, target)        # 计算 MSE
         │
         └─ 7. loss * pipe.scheduler.training_weight(t)  # 加权
                └─ 返回最终 loss
```


```python
# 位置: diffsynth/diffusion/loss.py:14-53
def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    # 1. 确定 timestep 采样范围
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    # 2. 均匀采样一个 timestep
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    # 3. 生成噪声并构造带噪样本
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)

    # 4. 构造训练目标
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    # 5. 模型预测
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    # 6. 计算加权 MSE loss
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss
```

---

## 3. Loss 计算原理


### 3.1 timestep 如何预先设置（训练 vs 推理）以及 `shift` 的作用

`FlowMatchSFTLoss` 里采样的 `timestep` 来自 `pipe.scheduler.timesteps`。这个数组并不是在 loss 内部“临时生成”的，而是**在进入训练/推理之前就由 scheduler 预先设定好**。

#### 3.1.1 训练模式：`set_timesteps(1000, training=True)`

训练初始化时会调用：

```python
# 位置: diffsynth/diffusion/training_module.py
pipe.scheduler.set_timesteps(1000, training=True)
```

这会让 `FlowMatchScheduler`：
- 生成长度为 1000 的 `sigmas` 与 `timesteps`
- 并在 `training=True` 时调用 `set_training_weight()` 预计算 `linear_timesteps_weights`

#### 3.1.2 推理模式：`set_timesteps(num_inference_steps, ..., shift=sigma_shift)`

推理端（`WanVideoPipeline.__call__`）会在每次调用时根据采样步数与配置生成 timesteps：

```python
# 位置: diffsynth/pipelines/wan_video.py
self.scheduler.set_timesteps(
    num_inference_steps,
    denoising_strength=denoising_strength,
    shift=sigma_shift,
)
```

> 注意：训练端没有显式传 `shift`，因此会使用 Wan 模板的默认 `shift=5`；推理端默认参数名叫 `sigma_shift`，本质也是同一个 `shift`。

#### 3.1.3 `shift` 到底改变了什么？

以 Wan 模板的实现为例（`FlowMatchScheduler.set_timesteps_wan`），核心逻辑是：

1) 先生成一个“线性”的 $\sigma$ 序列（受 `denoising_strength` 影响）：

$$
\sigma^{raw}_i=\text{linspace}(\sigma_{start},\sigma_{min},N+1)_{i},\quad i=0,\dots,N-1
$$

2) 再用 `shift` 做一次单调变换（代码里是一个分式）：

$$
\sigma_i=\frac{k\cdot \sigma^{raw}_i}{1+(k-1)\cdot \sigma^{raw}_i},\qquad k=\text{shift}.
$$

3) 最后把 $\sigma$ 映射成 “timestep 数值”：

$$
t_i=\sigma_i\cdot 1000.
$$

**直观效果（当 $k>1$）**：
- $\sigma_i$ 会整体“向 1 抬高”（同一个 $\sigma^{raw}$ 会变成更大的 $\sigma$），意味着采样/训练更偏向噪声更强的区间；
- 同时该变换在 $\sigma\approx 1$ 附近更“慢”、在 $\sigma\approx 0$ 附近更“快”，因此**离散网格在高噪声区间更密、低噪声区间更稀**（更强调早期高噪声步骤的分辨率）。

一个常用的判断方式是看导数（对上式求导）：

$$
\frac{d\sigma}{d\sigma^{raw}}=\frac{k}{(1+(k-1)\sigma^{raw})^2},
$$

当 $\sigma^{raw}\to 0$ 时导数趋近 $k$（变化更快、点更稀），当 $\sigma^{raw}\to 1$ 时导数趋近 $1/k$（变化更慢、点更密）。

#### 3.1.4 loss 里采样的到底是“索引”还是“数值”？

`FlowMatchSFTLoss` 先均匀采样一个 **索引** `timestep_id`，再取出 **数值** `timestep = timesteps[timestep_id]`：

- `timestep_id`：离散索引（在 `[0, len(timesteps))` 上均匀）
- `timestep`：连续数值（大致对应 $t=\sigma\cdot 1000$）

因此当 `shift` 改变了 $\sigma_i$ 的分布时，即使 `timestep_id` 是均匀的，**被采样到的噪声强度 $\sigma$ 也会随之改变分布**。

### 3.2 数学公式

#### 3.2.1 加噪公式（`add_noise`）

训练时我们先在 latent 空间构造带噪样本 $x_t$（对应代码里的 `inputs["latents"]`）：

$$
x_t=(1-\sigma_t)\cdot x_0+\sigma_t\cdot \varepsilon,\qquad \varepsilon\sim\mathcal N(0, I).
$$

其中：
- $x_0$：干净样本（Ground Truth 视频经 VAE 编码后的 latent，对应 `input_latents`）
- $\varepsilon$：标准高斯噪声（`torch.randn_like(input_latents)`）
- $\sigma_t$：timestep $t$ 对应的噪声强度（从 scheduler 的 `sigmas` 查表得到）
- $x_t$：t 时刻的带噪样本（送入模型的 noisy latent）

**直观理解**：$x_t$ 是 $x_0$ 与纯噪声 $\varepsilon$ 的线性插值：$\sigma_t=0$ 时 $x_t=x_0$，$\sigma_t=1$ 时 $x_t=\varepsilon$。

#### 3.2.2 训练目标（`training_target`）

本实现的训练目标（代码变量 `training_target`）定义为：

$$
v_{\text{target}}=\varepsilon-x_0.
$$

**为什么是这个形式？**  
把“从 $x_0$ 到 $\varepsilon$ 的直线路径”按噪声强度 $\sigma$ 参数化：

$$
x(\sigma)=(1-\sigma)\cdot x_0+\sigma\cdot \varepsilon.
$$

对 $\sigma$ 求导得到速度场（velocity field）：

$$
v=\frac{d x(\sigma)}{d\sigma}=-x_0+\varepsilon=\varepsilon-x_0.
$$

因此也可写成：

$$
x(\sigma)=x_0+\sigma\cdot v.
$$

反解得到：

$$
x_0=x(\sigma)-\sigma\cdot v.
$$

推理/采样时的更新（对应 `FlowMatchScheduler.step`）就是沿这个速度场做“离散积分”：

$$
x(\sigma')=x(\sigma)+(\sigma'-\sigma)\cdot v.
$$

由于采样时 $\sigma$ 递减（$\sigma'<\sigma$），所以 $(\sigma'-\sigma)<0$，更新方向等价于“沿 $-v$ 逼近 $x_0$”。

#### 3.2.3 损失函数

将模型输出记为 $v_{\text{pred}}=\text{DiT}(x_t,\text{prompt},t)$（代码里变量名叫 `noise_pred`），则：

$$
L=w(t)\cdot \operatorname{MSE}(v_{\text{pred}}, v_{\text{target}})
=w(t)\cdot \operatorname{MSE}\bigl(v_{\text{pred}},\varepsilon-x_0\bigr).
$$

其中 $w(t)$ 是 timestep 权重（下一节解释其构造与含义）。



---

## 4. Timestep 加权机制

### 4.1 为什么需要加权？

**问题**：不同 timestep 的学习难度不同。

```
timestep ≈ 0   (σ ≈ 0):  xt ≈ x0，几乎无噪声，预测太简单
timestep ≈ 500 (σ ≈ 0.5): xt 既有信号也有噪声，最有学习价值
timestep ≈ 1000 (σ ≈ 1): xt ≈ ε，几乎纯噪声，信号太弱
```

**解决方案**：使用高斯权重曲线，中间 timestep 权重高，两端权重低。

### 4.2 权重计算公式

```python
# 位置: diffsynth/diffusion/flow_match.py:141-160
def set_training_weight(self):
    steps = 1000
    x = self.timesteps                                    # [t_0, t_1, ..., t_n]
    y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)   # 高斯曲线
    y_shifted = y - y.min()                               # 平移到非负
    bsmntw_weighing = y_shifted * (steps / y_shifted.sum())  # 归一化

    if len(self.timesteps) != 1000:
        bsmntw_weighing = bsmntw_weighing * (len(self.timesteps) / steps)
        bsmntw_weighing = bsmntw_weighing + bsmntw_weighing[1]

    self.linear_timesteps_weights = bsmntw_weighing
```

#### 4.2.1 为什么能实现“中间高、两端低”？

核心原因是：权重曲线本质上是一个**以 $x\approx 500$ 为中心的钟形函数**，再经过“平移到非负 + 归一化”得到最终权重。

1) 先看高斯形状（`y = exp(...)`）  
代码里定义（把 `steps=1000` 代入）：

$$
y(x)=\exp\left(-2\left(\frac{x-500}{1000}\right)^2\right).
$$

它在 $x=500$ 处取得最大值 $y(500)=1$，并且关于 $x=500$ 对称；越靠近两端（$x$ 越远离 500），$y(x)$ 越小。

2) 再看“平移到非负”（`y_shifted = y - y.min()`）  
`y.min()` 出现在两端附近（$x$ 最远离 500 的位置），因此：

$$
\tilde{y}(x)=y(x)-\min_x y(x)\ge 0,
$$

并且两端（最小值处）会被压到 0，中间仍为正数 ⇒ **两端低、中间高** 的形状被进一步强化。

3) 最后是“归一化”（`* (steps / y_shifted.sum())`）  
这一步只是把整体乘上一个常数，使所有权重的和满足：

$$
w_i=\tilde{y}_i\cdot \frac{1000}{\sum_j \tilde{y}_j}
\quad\Rightarrow\quad
\sum_i w_i = 1000.
$$

归一化不会改变相对形状（谁高谁低不变），只改变整体尺度。因此最终 `linear_timesteps_weights` 仍然保持“中间高、两端低”。

> 小提示：这里的 $x$ 不是“索引 i”，而是 `self.timesteps` 的数值（大致在 $[0,1000]$）。因此“中间”更准确地说是 **噪声强度处于中等区间（$\sigma\approx 0.5$）的位置**，而不是数组下标的正中间。

#### 4.2.2 `len(self.timesteps) != 1000` 分支在做什么？

这一段注释里也写了 “empirical formula”，更像是为了兼容“训练步数不是 1000”的场景（例如某些 loss 里用很少的 steps 也想要一个不至于全为 0 的权重表）：

- `bsmntw_weighing * (len/steps)`：把“总和约为 1000”的权重，缩放到“总和约为 len(self.timesteps)”的量级；
- `+ bsmntw_weighing[1]`：整体抬高一个常数，避免出现大量 0 权重。

对 Wan 的 SFT 训练（通常 `set_timesteps(1000, training=True)`），这条分支一般不会触发。

### 4.3 权重曲线可视化

```
权重 w(t)
    │
2.0 ┤                    ▄▄▄▄▄▄
    │                 ▄▄█      █▄▄
1.5 ┤              ▄▄█            █▄▄
    │           ▄▄█                  █▄▄
1.0 ┤        ▄▄█                        █▄▄
    │     ▄▄█                              █▄▄
0.5 ┤  ▄▄█                                    █▄▄
    │▄█                                          █▄
0.0 ┼────┬────┬────┬────┬────┬────┬────┬────┬────┬────►
    0   100  200  300  400  500  600  700  800  900 1000
                        timestep t

    ◄──── 权重低 ────►◄──── 权重高 ────►◄──── 权重低 ────►
         (易学习)         (高信号)           (低信号)
```

### 4.4 权重查询

```python
# 位置: diffsynth/diffusion/flow_match.py:226-230
def training_weight(self, timestep):
    timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
    weights = self.linear_timesteps_weights[timestep_id]
    return weights
```

---

## 5. 视频 SFT 训练细节

### 5.1 视频 Latent 的形状与加噪方式

**核心问题**：加噪是针对整个视频还是单独某一帧？

**答案**：整个视频的所有帧**同时加噪**，使用同一个噪声强度 σ。

```python
# input_latents 形状：[B, C, T, H, W]
# 例如：[1, 16, 13, 60, 104]
#       B=1, C=16(latent channels), T=13(时间帧), H=60, W=104

noise = torch.randn_like(inputs["input_latents"])  # 同形状的噪声
# noise 形状也是 [1, 16, 13, 60, 104]

# 所有帧用同一个 sigma 加噪
xt = (1 - σ) * x0 + σ * ε  # 对整个 [B,C,T,H,W] tensor 操作
```

```
┌─────────────────────────────────────────────────────────────────┐
│  标准 Flow Matching SFT 加噪方式                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  x0 (干净视频):                                                  │
│  ┌─────┬─────┬─────┬─────┬─────┐                               │
│  │ f0  │ f1  │ f2  │ ... │ fT  │  所有帧都是干净的               │
│  └─────┴─────┴─────┴─────┴─────┘                               │
│           ↓  同一个 σ 同时加噪                                   │
│  xt (带噪视频):                                                  │
│  ┌─────┬─────┬─────┬─────┬─────┐                               │
│  │ f0' │ f1' │ f2' │ ... │ fT' │  所有帧噪声程度相同 σ           │
│  └─────┴─────┴─────┴─────┴─────┘                               │
│                                                                 │
│  特点：没有"干净历史帧"的概念，所有帧处于同一噪声水平              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 模型预测目标

**核心问题**：模型预测的是什么？需要预测每一帧对应的 v 吗？

**答案**：模型预测整个视频所有帧的 velocity field `v = ε - x0`，一次 forward 输出完整视频的预测。

```python
# 模型输入
noise_pred = pipe.model_fn(
    **models,
    latents=xt,           # [B, C, T, H, W] - 带噪的完整视频
    timestep=timestep,    # 标量 - 当前时间步
    ...
)

# 模型输出
# noise_pred 形状：[B, C, T, H, W] - 与输入相同
# 每个位置 (b, c, t, h, w) 都有一个预测值
```

从 `unpatchify` 可以看到输出重构过程：

```python
# 位置: diffsynth/models/wan_video_dit.py:346-351
def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
    return rearrange(
        x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
        f=grid_size[0],  # 时间维
        h=grid_size[1],  # 高度
        w=grid_size[2],  # 宽度
        x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
    )
```

### 5.3 Attention 机制

**核心问题**：是否需要干净的历史帧？是否需要特殊的 attention map？

**答案**：Wan 使用**全局 3D Self-Attention（非因果）**，不需要干净历史帧或特殊 mask。

```python
# 位置: diffsynth/models/wan_video_dit.py:139-145
class SelfAttention(nn.Module):
    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)  # 3D RoPE 位置编码
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)  # 全局注意力，无 causal mask
        return self.o(x)
```

```
┌─────────────────────────────────────────────────────────────────┐
│  Wan 3D Self-Attention                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Patchify 后的 token 序列：                                      │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐            │
│  │t0 │t0 │t0 │t1 │t1 │t1 │t2 │t2 │t2 │...│tT │tT │            │
│  │p0 │p1 │p2 │p0 │p1 │p2 │p0 │p1 │p2 │   │pN │pN │            │
│  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘            │
│    ↑                                                            │
│    所有 token 在 self-attention 中互相可见                        │
│                                                                 │
│  Attention 特点：                                                │
│  ✓ 全局注意力（Non-causal）：每个 token 可以看到所有其他 token    │
│  ✓ 3D RoPE：位置编码包含 (t, h, w) 三维信息                      │
│  ✗ 不是因果注意力：未来帧可以看到过去帧，反之亦然                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 TI2V（Text-Image-to-Video）的特殊处理

对于 TI2V 任务，第一帧（input_image）有特殊处理：

```python
# 位置: diffsynth/pipelines/wan_video.py:655-662
def process(self, pipe, input_image, latents, ...):
    if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
        return {}
    # 第一帧直接用 VAE 编码的干净 latent 替换
    z = pipe.vae.encode([image], ...)
    latents[:, :, 0:1] = z  # 替换第一帧！
    return {"latents": latents, ...}
```

```
┌─────────────────────────────────────────────────────────────────┐
│  TI2V 训练的特殊情况                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  如果 fuse_vae_embedding_in_latents=True：                      │
│                                                                 │
│  latents[:, :, 0:1] 被替换为 input_image 的 VAE 编码            │
│                                                                 │
│  ┌───────┬─────┬─────┬─────┬─────┐                             │
│  │ f0    │ f1' │ f2' │ ... │ fT' │                             │
│  │(干净) │(加噪)│(加噪)│     │(加噪)│                             │
│  └───────┴─────┴─────┴─────┴─────┘                             │
│     ↑                                                           │
│     第一帧保持干净（条件图像）                                     │
│                                                                 │
│  这种情况下，模型学习的是：                                       │
│  "给定干净的第一帧 + 带噪的后续帧，预测整个视频的 velocity"        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.5 训练方式对比总结

| 问题 | 标准 SFT | TI2V (fuse_vae_embedding) |
|------|----------|---------------------------|
| 加噪范围 | 所有帧 | 除第一帧外的所有帧 |
| 干净历史帧 | 无 | 第一帧保持干净 |
| Attention | 全局 3D（非因果） | 全局 3D（非因果） |
| 预测目标 | 整个视频的 v | 整个视频的 v |

> **关键点**：标准 Flow Matching SFT **不需要** autoregressive 式的因果注意力或干净历史帧。模型在训练时看到的是整个视频同时处于某个噪声水平，学习从该噪声状态恢复整个视频。这与 World Model 类的 autoregressive 视频生成不同。

---

### 5.6 完整训练步骤流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        单步训练 (Single Training Step)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  输入数据                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  video: [49 frames, 480×832, RGB]                               │   │
│  │  prompt: "A cat walking in the garden"                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  Pipeline Units 预处理                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. VAE Encoder: video → input_latents [1,16,13,60,104]         │   │
│  │  2. Text Encoder: prompt → prompt_emb [1,512,4096]              │   │
│  │  3. (可选) Image Encoder: first_frame → image_emb               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  FlowMatchSFTLoss 计算                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                 │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │   │
│  │  │x0 (input_    │    │ ε (noise)    │    │ t (timestep) │      │   │
│  │  │   latents)   │    │ [1,16,13,    │    │ t≈500        │      │   │
│  │  │ [1,16,13,    │    │  60,104]     │    │ σ≈0.50       │      │   │
│  │  │  60,104]     │    │              │    │              │      │   │
│  │  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │   │
│  │         │                   │                   │               │   │
│  │         └─────────┬─────────┴─────────┬─────────┘               │   │
│  │                   │                   │                         │   │
│  │                   ▼                   ▼                         │   │
│  │         ┌─────────────────┐  ┌─────────────────┐               │   │
│  │         │   add_noise     │  │ training_target │               │   │
│  │         │ xt=(1-σ)x0+σε   │  │ target = ε - x0 │               │   │
│  │         └────────┬────────┘  └────────┬────────┘               │   │
│  │                  │                    │                         │   │
│  │                  ▼                    │                         │   │
│  │         ┌─────────────────┐           │                         │   │
│  │         │   DiT Model     │           │                         │   │
│  │         │  (with LoRA)    │           │                         │   │
│  │         │                 │           │                         │   │
│  │         │ Input: xt, t,   │           │                         │   │
│  │         │   prompt_emb    │           │                         │   │
│  │         │                 │           │                         │   │
│  │         │ Output:         │           │                         │   │
│  │         │   noise_pred    │           │                         │   │
│  │         └────────┬────────┘           │                         │   │
│  │                  │                    │                         │   │
│  │                  └──────────┬─────────┘                         │   │
│  │                             │                                   │   │
│  │                             ▼                                   │   │
│  │                  ┌─────────────────┐                            │   │
│  │                  │    MSE Loss     │                            │   │
│  │                  │ ||pred-target||²│                            │   │
│  │                  └────────┬────────┘                            │   │
│  │                           │                                     │   │
│  │                           ▼                                     │   │
│  │                  ┌─────────────────┐                            │   │
│  │                  │ training_weight │                            │   │
│  │                  │   loss *= w(t)  │                            │   │
│  │                  │  w(500)≈2.08    │                            │   │
│  │                  └────────┬────────┘                            │   │
│  │                           │                                     │   │
│  └───────────────────────────┼─────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  梯度更新                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. accelerator.backward(loss)   # 反向传播                      │   │
│  │  2. optimizer.step()             # 更新 LoRA 参数                │   │
│  │     └─ 只有 ~0.26M LoRA 参数被更新，原始 5B 参数冻结              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 与 DDPM 的详细对比

> **说明**：以下对比以经典 DDPM 的 ε-prediction 参数化为参照。
> 实际上 DDPM 也有 v-prediction、x0-prediction 等变体，采样也可通过 DDIM 等加速。

### 6.1 训练目标对比

```
DDPM (经典 ε-prediction):
  ┌─────────────────────────────────────────────────────┐
  │  加噪: xt = √(ᾱt) * x0 + √(1-ᾱt) * ε                │
  │  目标: 预测噪声 ε                                   │
  │  Loss: ||ε_pred - ε||²                              │
  └─────────────────────────────────────────────────────┘

Flow Matching (本实现):
  ┌─────────────────────────────────────────────────────┐
  │  加噪: xt = (1 - σt) * x0 + σt * ε                  │
  │  目标: 预测速度场 v = ε - x0                        │
  │  Loss: ||v_pred - (ε - x0)||² * w(t)               │
  └─────────────────────────────────────────────────────┘
```

### 6.2 采样路径对比

```
DDPM 逆过程（经典为随机 Markov chain，DDIM 等可加速为确定性）:

    x_T ──┐
          │ 逐步去噪
    x_t ──┼──┐
              │
    x_s ──────┼──┐
                  │
    x_0 ──────────┘

Flow Matching（ODE，确定性直线路径）:

    x_T ─────────────────┐
                          ╲  直线路径
    x_t ──────────────────╲─┐
                             ╲
    x_s ──────────────────────╲─┐
                                 ╲
    x_0 ──────────────────────────┘
```
