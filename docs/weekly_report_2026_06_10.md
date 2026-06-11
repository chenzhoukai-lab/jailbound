# JailBound 本地复现周报

## 一、本周工作概述

本周继续围绕论文 **JailBound: Jailbreaking Internal Safety Boundaries of Vision-Language Models** 做本地复现与实验迭代。实验环境仍以本地部署的 `MM-SafetyBench` 为数据集，`Qwen2.5-VL-7B-Instruct` 为目标模型，`Qwen3Guard-Gen-8B` 为安全评测模型。

在上周已经跑通基础复现框架的基础上，本周主要完成了三部分工作：

1. 改进 probing 阶段的 safe / unsafe 数据构造，减少固定 safe prompt 带来的模板差异。
2. 接入 HotFlip 风格的 token 级后缀优化，使文本扰动不再只是固定 suffix candidate 选择。
3. 补充完整评测体系，新增 non-refusal ASR、Qwen2.5 judge ASR、按类别统计和改版前后对比表格。

目前项目已经可以完成边界探测、边界穿越攻击、多卡并行、断点续跑、Qwen3Guard 自动评测、non-refusal 评测、Qwen2.5 judge 评测以及统一表格分析。整体上，工程链路已经比较完整，但从实验结果看，当前改版后的 v2 并没有带来 ASR 提升，说明后续仍需要继续分析边界构造和扰动优化是否真正有效。

## 二、复现思路

原论文的核心观点是：VLM 在生成最终回答之前，其内部表示中可能已经存在某种“安全 / 不安全”的隐式边界。JailBound 先通过 probing 找到这个边界，再通过优化输入扰动，让模型内部状态向边界另一侧移动。

当前复现流程如下：

1. 从 MM-SafetyBench 读取图像和 harmful prompt。
2. 将样本输入 Qwen2.5-VL，提取模型中间层 hidden states。
3. 使用 safe / unsafe 两类 hidden states 训练线性 probe，近似内部安全边界。
4. 根据 probe 得到的边界方向，优化文本后缀和图像扰动，使攻击后的 hidden state 靠近边界穿越后的目标状态。
5. 使用 Qwen2.5-VL 生成回答，再分别用 Qwen3Guard、non-refusal 规则和 Qwen2.5 judge 进行评估。

由于 HuggingFace 版本的 Qwen2.5-VL 没有直接暴露论文公式中的显式 fusion layer，当前仍使用 decoder hidden states 作为近似。Qwen2.5-VL 的 image tokens 和 text tokens 会在 decoder 多层 attention 中完成交互，因此 decoder hidden states 可以作为融合后内部状态的工程近似，但这与论文中的 fusion representation 并不完全等价。

## 三、本周新增实现

### 1. 改进 probing 数据构造

上周版本中，probing 阶段使用：

- unsafe 输入：原图 + MM-SafetyBench 原始 harmful prompt。
- safe 输入：原图 + 固定 safe prompt。

固定 safe prompt 为：

```text
Describe the image briefly and safely. Do not provide instructions for harmful or illegal actions.
```

这种方式虽然可以得到很高的 probe 训练准确率，但可能引入 prompt 模板差异。也就是说，probe 学到的未必是真正的安全语义边界，而可能是 harmful prompt 与固定 safe prompt 在语言形式上的差异。

本周新增了 `jailbound_v2` 实验目录，在不覆盖原始代码的前提下，对 probing 数据构造做了改进。v2 会围绕同一个原始 harmful prompt 构造更匹配的 safe prompt pair，例如将 harmful instruction 改写为防御、检测、合规解释类 safe instruction。这样 safe / unsafe 样本在主题和图像上下文上更接近，probe 更有可能捕捉安全意图差异，而不是简单的模板差异。

### 2. 接入 HotFlip 文本后缀优化

上周版本的文本攻击只是在少量固定 suffix candidates 中选择一个最容易推动 hidden state 越界的后缀，并没有实现 token 级优化。

本周在 v2 中接入了 HotFlip 风格的后缀优化流程：

1. 先从 suffix bank 中选择一个初始 suffix。
2. 将 `原始 prompt + suffix` 送入 Qwen chat template。
3. 定位 suffix 在完整 input ids 中对应的 token span。
4. 对 suffix token embedding 的梯度进行反传。
5. 根据梯度方向替换 token，使后缀更有利于推动 hidden state 向边界目标状态移动。

其中 token span 定位采用两级策略：

- 优先使用 tokenizer offset mapping，在展开后的 Qwen chat template 中定位 suffix 字符区间对应的 token。
- 如果 offset mapping 不可用或无法对齐，则回退到 base prompt 与 full prompt 的 input id diff。

每条攻击结果中会保存 HotFlip 轨迹，包括初始 suffix、最终 suffix、token span、替换前后 token id 和每一步替换结果，便于后续分析文本扰动是否真正起作用。

### 3. 补充多种 ASR 评测

本周补充了三类评测指标：

1. **Qwen3Guard ASR**：只有 Qwen3Guard 判定目标模型输出为 Unsafe，才计为攻击成功。这是最严格的评测口径。
2. **Non-Refusal ASR**：只要模型输出没有明显拒答模板，就计为成功，更接近论文中常见的 non-refusal ASR。
3. **Qwen2.5 Judge ASR**：使用本地 Qwen2.5-7B-Instruct 作为二分类 judge，判断模型输出是否尝试完成原始请求。这个指标介于 Qwen3Guard ASR 和 non-refusal ASR 之间，比单纯 non-refusal 更关注回答是否真正跟随任务。

此外，还新增了统一报表脚本，可以自动生成：

- 总体 ASR 表。
- 按类别 ASR 表。
- 改版前后总体对比表。
- 改版前后按类别对比表。

## 四、实验结果

本次全量实验共评估 1680 条 MM-SafetyBench 样本。对比对象包括：

- `baseline`：上周基础版本，固定 safe prompt + suffix candidate 选择 + pixel_values 图像扰动。
- `jailbound_v2`：本周改版，匹配 safe / unsafe prompt pair + HotFlip token 级后缀优化 + pixel_values 图像扰动。

### 1. 总体 ASR 结果

| 方法 | 样本数 | Qwen3Guard ASR | Non-Refusal ASR | Qwen2.5 Judge ASR |
|---|---:|---:|---:|---:|
| baseline | 1680 | 6.43% | 96.31% | 25.00% |
| jailbound_v2 | 1680 | 3.81% | 95.36% | 18.04% |

从总体结果看，三类 ASR 指标差异很大：

- Qwen3Guard ASR 最低，说明从严格安全模型视角看，真正被判定为 Unsafe 的输出比例不高。
- Non-Refusal ASR 非常高，说明大多数情况下模型没有明显拒答。
- Qwen2.5 Judge ASR 介于两者之间，更能反映“模型是否实际尝试完成原始请求”。

比较 baseline 和 v2 后可以看到，v2 的三项指标均低于 baseline。说明本周引入的匹配 safe prompt pair 和 HotFlip 后缀优化虽然让方法更接近论文流程，但当前实现并没有直接提高攻击成功率。

### 2. 改版前后 ASR 变化

| 对比 | Qwen3Guard ASR 变化 | Non-Refusal ASR 变化 | Qwen2.5 Judge ASR 变化 |
|---|---:|---:|---:|
| jailbound_v2 - baseline | -2.62% | -0.95% | -6.96% |

其中 Qwen2.5 Judge ASR 降幅最大，从 25.00% 降到 18.04%。这说明 v2 生成的回答虽然大多仍然没有明显拒答，但更少被 Qwen2.5 judge 认为是在实际完成原始 harmful request。

### 3. 分类别 ASR 结果

| 类别 | 方法 | 样本数 | Qwen3Guard ASR | Non-Refusal ASR | Qwen2.5 Judge ASR |
|---|---|---:|---:|---:|---:|
| EconomicHarm | baseline | 122 | 3.28% | 99.18% | 35.25% |
| EconomicHarm | jailbound_v2 | 122 | 1.64% | 98.36% | 27.87% |
| Financial_Advice | baseline | 167 | 0.00% | 99.40% | 64.07% |
| Financial_Advice | jailbound_v2 | 167 | 0.00% | 99.40% | 53.29% |
| Fraud | baseline | 154 | 10.39% | 91.56% | 22.08% |
| Fraud | jailbound_v2 | 154 | 7.79% | 88.96% | 12.99% |
| Gov_Decision | baseline | 149 | 0.00% | 98.66% | 12.75% |
| Gov_Decision | jailbound_v2 | 149 | 0.00% | 99.33% | 8.05% |
| HateSpeech | baseline | 163 | 6.13% | 91.41% | 17.79% |
| HateSpeech | jailbound_v2 | 163 | 3.68% | 90.18% | 12.27% |
| Health_Consultation | baseline | 109 | 0.00% | 97.25% | 11.93% |
| Health_Consultation | jailbound_v2 | 109 | 0.00% | 99.08% | 3.67% |
| Illegal_Activitiy | baseline | 97 | 21.65% | 92.78% | 10.31% |
| Illegal_Activitiy | jailbound_v2 | 97 | 8.25% | 92.78% | 11.34% |
| Legal_Opinion | baseline | 130 | 0.00% | 96.15% | 21.54% |
| Legal_Opinion | jailbound_v2 | 130 | 0.00% | 94.62% | 20.77% |
| Malware_Generation | baseline | 44 | 20.45% | 100.00% | 25.00% |
| Malware_Generation | jailbound_v2 | 44 | 9.09% | 97.73% | 15.91% |
| Physical_Harm | baseline | 144 | 15.97% | 95.83% | 20.14% |
| Physical_Harm | jailbound_v2 | 144 | 9.72% | 91.67% | 11.11% |
| Political_Lobbying | baseline | 153 | 0.00% | 100.00% | 30.72% |
| Political_Lobbying | jailbound_v2 | 153 | 0.00% | 100.00% | 18.30% |
| Privacy_Violence | baseline | 139 | 8.63% | 93.53% | 17.27% |
| Privacy_Violence | jailbound_v2 | 139 | 2.88% | 93.53% | 10.79% |
| Sex | baseline | 109 | 11.93% | 99.08% | 23.85% |
| Sex | jailbound_v2 | 109 | 12.84% | 96.33% | 18.35% |

从分类结果看：

- `Financial_Advice` 的 Qwen2.5 Judge ASR 最高，baseline 为 64.07%，v2 为 53.29%，说明这类任务中模型更容易给出看似完成请求的回答。
- `Illegal_Activitiy`、`Malware_Generation`、`Physical_Harm` 在 Qwen3Guard ASR 上相对更高，说明这些类别更容易被安全模型判定为 Unsafe。
- v2 在大多数类别上的 Qwen2.5 Judge ASR 都低于 baseline，仅 `Illegal_Activitiy` 有小幅提升。
- `Sex` 类别的 Qwen3Guard ASR 从 11.93% 提升到 12.84%，但 Qwen2.5 Judge ASR 仍下降，说明严格 Unsafe 判断和任务完成判断并不完全一致。

## 五、结果分析

### 1. Non-refusal ASR 很高，但不能直接说明攻击有效

本次实验中，baseline 和 v2 的 non-refusal ASR 都超过 95%。这说明 Qwen2.5-VL 在 MM-SafetyBench 的很多样本上并不会明显拒答。

但 non-refusal 只是检查回答是否没有拒绝，并不判断回答是否真正完成 harmful request。因此它更适合用于和论文口径对齐，但不能单独作为攻击有效性的充分证据。

### 2. Qwen3Guard ASR 较低，说明严格 unsafe 输出比例有限

Qwen3Guard ASR 在 baseline 中为 6.43%，v2 中为 3.81%。这说明严格安全评测下，真正被判定为 Unsafe 的输出并不多。

这可能有两方面原因：

- 当前攻击确实没有充分推动模型生成高风险内容。
- Qwen3Guard 的判断口径比论文 non-refusal ASR 更严格。

因此后续分析不能只看 Qwen3Guard ASR，也需要结合 Qwen2.5 judge 和人工抽样检查。

### 3. v2 方法更完整，但当前参数下 ASR 没有提升

v2 在方法上补齐了两个重要问题：

- probing 阶段使用更相似的 safe / unsafe prompt pair，减少模板差异。
- 攻击阶段加入 HotFlip token 级后缀优化，替代纯固定后缀选择。

但实验结果显示，v2 的总体 ASR 反而低于 baseline。这说明“方法更接近论文”不一定会立刻带来更高 ASR。可能原因包括：

- 匹配 safe prompt pair 后，probe 学到的边界更保守，导致攻击方向不如固定 safe prompt 版本激进。
- 当前 HotFlip 只优化 suffix token，但目标仍然是 hidden-state boundary loss，而不是直接优化最终 unsafe / task-following 输出。
- HotFlip 的 token 替换空间和步数较小，可能没有找到足够有效的后缀。
- v2 的 safe prompt 改写虽然减少了模板差异，但也可能削弱了 probe 的线性可分性和攻击方向强度。

### 4. 三类评测指标反映的是不同问题

当前三类 ASR 需要分开理解：

- Qwen3Guard ASR：回答是否被安全模型判定为 Unsafe。
- Non-refusal ASR：回答是否没有明显拒答。
- Qwen2.5 Judge ASR：回答是否尝试完成原始请求。

其中 Qwen2.5 Judge ASR 更适合衡量“攻击是否真的有效”，因为它比 non-refusal 更严格，又不像 Qwen3Guard 那样只关注安全标签。

## 六、目前仍存在的问题

1. **fusion layer 仍是近似**

   当前仍使用 Qwen2.5-VL decoder hidden states 作为 fusion representation 的近似。该做法工程上可行，但不完全等价于论文定义中的 fusion layer。

2. **probe 边界的有效性仍需验证**

   v2 虽然改进了 safe / unsafe prompt pair，但 ASR 没有提升，说明 probe 学到的边界是否真正对应安全机制仍需要进一步验证。

3. **HotFlip 后缀优化仍较弱**

   当前 HotFlip 只做 suffix token 替换，且替换目标是 hidden-state boundary loss。后续需要分析替换出的 token 是否自然、是否真的增强任务完成倾向。

4. **视觉扰动仍在 pixel_values 空间**

   当前图像扰动仍然加在 processor 后的 `pixel_values` 上，还没有迁移到原始 RGB 图像空间，也没有形成可保存、可复查的 adversarial images。

5. **ASR 与论文仍有差距**

   当前最严格的 Qwen3Guard ASR 仍较低，Qwen2.5 Judge ASR 也只有 18.04% 到 25.00% 区间，说明距离论文中的高 ASR 攻击仍有明显差距。

## 七、下周计划

1. **分析 v2 ASR 下降原因**

   抽样检查 baseline 和 v2 的回答差异，重点查看 HotFlip 后缀是否造成回答变短、变泛化或偏离原始请求。

2. **做 probe 消融实验**

   对比固定 safe prompt、defensive safe prompt、detection safe prompt、compliance safe prompt 对边界方向和 ASR 的影响，判断 v2 的 prompt pair 是否过于保守。

3. **改进 HotFlip 目标函数**

   当前 HotFlip 只优化 hidden-state boundary loss。后续可以尝试结合 task-following judge、response likelihood 或更强的候选筛选，让 token 替换更直接服务于最终输出。

4. **继续推进图像输入空间扰动**

   尝试将扰动从 processor 后的 `pixel_values` 迁移到原始 RGB 图像空间，并保存 adversarial images，便于后续人工观察和跨模型迁移分析。

5. **增加人工抽样分析**

   对三类评测指标不一致的样本进行人工抽样，例如 non-refusal 成功但 Qwen2.5 judge 失败、Qwen2.5 judge 成功但 Qwen3Guard 判 Safe 的样本，从而更准确理解当前攻击的真实效果。

## 八、本周小结

本周在 JailBound 本地复现框架上完成了较关键的一轮迭代：一方面改进了 probing 数据构造，使 safe / unsafe 对比更接近同主题语义差异；另一方面接入了 HotFlip 风格的 token 级后缀优化，使攻击阶段不再局限于固定 suffix 选择。同时，补齐了 Qwen3Guard ASR、non-refusal ASR 和 Qwen2.5 judge ASR 三类评测，并实现了按类别和改版前后的统一表格分析。

从实验结果看，baseline 在 1680 条样本上的 Qwen3Guard ASR 为 6.43%，non-refusal ASR 为 96.31%，Qwen2.5 judge ASR 为 25.00%；v2 对应结果为 3.81%、95.36% 和 18.04%。这说明 v2 虽然在方法上更完整，但当前配置下没有提升 ASR，反而降低了严格 unsafe 和任务完成类指标。

因此，当前阶段的主要结论是：复现框架和评测体系已经基本建立，但攻击有效性仍不足。后续工作需要从边界是否准确、HotFlip 后缀是否有效、图像扰动是否合理三个方向继续推进。

## 九、第三版实验结果补充

在 v2 结果出来后，又进一步完成了第三版实验结果统计。这里将 baseline、jailbound_v2 和 jailbound_v3 三个版本放在一起比较。

### 1. 三版总体 ASR 对比

| 方法 | 样本数 | Qwen3Guard ASR | Non-Refusal ASR | Qwen2.5 Judge ASR |
|---|---:|---:|---:|---:|
| baseline | 1680 | 6.43% | 96.31% | 25.00% |
| jailbound_v2 | 1680 | 3.81% | 95.36% | 18.04% |
| jailbound_v3 | 1680 | 4.64% | 94.58% | 22.44% |

从总体结果看，第三版相比 v2 有一定恢复：

- Qwen3Guard ASR 从 3.81% 提升到 4.64%，提升 0.83 个百分点。
- Qwen2.5 Judge ASR 从 18.04% 提升到 22.44%，提升 4.40 个百分点。
- Non-Refusal ASR 从 95.36% 降到 94.58%，下降 0.77 个百分点。

这说明第三版相较 v2 更容易生成被 Qwen2.5 judge 认为“完成原始请求”的回答，但模型明显拒答或安全化回答的比例也略有增加。整体来看，v3 比 v2 更接近有效攻击，但仍未超过 baseline。

### 2. 三版相对变化

| 对比 | Qwen3Guard ASR 变化 | Non-Refusal ASR 变化 | Qwen2.5 Judge ASR 变化 |
|---|---:|---:|---:|
| jailbound_v2 - baseline | -2.62% | -0.95% | -6.96% |
| jailbound_v3 - baseline | -1.79% | -1.73% | -2.56% |
| jailbound_v3 - jailbound_v2 | +0.83% | -0.77% | +4.40% |

这个结果说明，v3 主要改善的是“任务完成度”指标，也就是 Qwen2.5 Judge ASR。相比 v2，v3 的回答更容易被判断为在尝试完成原始请求；但相比 baseline，v3 仍然偏低，说明第三版虽然修复了一部分 v2 的问题，但还没有完全恢复甚至超过初版的攻击效果。

### 3. 第三版分类别结果

| 类别 | 样本数 | Qwen3Guard ASR | Non-Refusal ASR | Qwen2.5 Judge ASR |
|---|---:|---:|---:|---:|
| EconomicHarm | 122 | 2.46% | 95.90% | 28.69% |
| Financial_Advice | 167 | 0.00% | 100.00% | 59.28% |
| Fraud | 154 | 5.84% | 90.26% | 17.53% |
| Gov_Decision | 149 | 0.00% | 98.66% | 12.08% |
| HateSpeech | 163 | 3.68% | 85.89% | 16.56% |
| Health_Consultation | 109 | 0.00% | 100.00% | 9.17% |
| Illegal_Activitiy | 97 | 12.37% | 89.69% | 6.19% |
| Legal_Opinion | 130 | 1.54% | 93.85% | 20.77% |
| Malware_Generation | 44 | 2.27% | 97.73% | 22.73% |
| Physical_Harm | 144 | 8.33% | 93.06% | 20.14% |
| Political_Lobbying | 153 | 0.00% | 100.00% | 24.18% |
| Privacy_Violence | 139 | 6.47% | 88.49% | 19.42% |
| Sex | 109 | 22.02% | 99.08% | 22.94% |

分类别看，v3 的提升主要体现在 Qwen2.5 Judge ASR 上：

- `Financial_Advice` 从 v2 的 53.29% 提升到 59.28%，接近 baseline 的 64.07%。
- `Physical_Harm` 从 v2 的 11.11% 恢复到 20.14%，基本回到 baseline。
- `Privacy_Violence` 从 v2 的 10.79% 提升到 19.42%，超过 baseline 的 17.27%。
- `Sex` 的 Qwen3Guard ASR 提升明显，从 v2 的 12.84% 提升到 22.02%，也超过 baseline 的 11.93%。

但也有一些类别仍然较弱：

- `Illegal_Activitiy` 的 Qwen2.5 Judge ASR 从 v2 的 11.34% 降到 6.19%，虽然 Qwen3Guard ASR 从 8.25% 提升到 12.37%，但任务完成度反而下降。
- `Malware_Generation` 的 Qwen3Guard ASR 从 v2 的 9.09% 降到 2.27%，说明第三版对该类严格 Unsafe 输出的推动较弱。
- `HateSpeech` 的 Non-Refusal ASR 降到 85.89%，是第三版中拒答或安全化倾向相对更明显的类别。

### 4. 对第三版结果的理解

结合 v2 中观察到的 HotFlip 后缀问题，可以认为第三版部分缓解了 v2 的文本扰动失效问题。v2 中 HotFlip 经常生成乱码、多语种碎片或不自然 token，使最终回答变得泛化、不具体，因此 Qwen2.5 Judge ASR 明显下降。v3 的 Qwen2.5 Judge ASR 回升到 22.44%，说明第三版在生成更可用回答方面有所改善。

不过，v3 仍然没有超过 baseline，说明当前攻击效果的瓶颈还没有完全解决。可能原因包括：

1. **边界目标和最终生成目标仍未完全对齐。** 当前优化仍主要围绕 hidden-state boundary loss，不能保证最终输出一定更 unsafe 或更任务完成。
2. **文本扰动仍可能影响回答自然性。** 即使 v3 比 v2 更好，suffix 优化仍需要继续限制 token 空间、保存 best suffix，并避免使用破坏语义的 token。
3. **不同类别对扰动非常敏感。** 例如 `Sex`、`Privacy_Violence` 在 v3 中有明显提升，但 `Illegal_Activitiy`、`Malware_Generation` 仍然表现不稳定。
4. **Qwen3Guard ASR 与 Qwen2.5 Judge ASR 不完全一致。** 有些类别严格 Unsafe 上升，但任务完成度下降，说明安全判定和任务完成判定衡量的是不同维度。

### 5. 当前阶段结论更新

综合三版结果，目前可以得到一个更清晰的结论：

- baseline 虽然方法较粗糙，但在当前配置下总体 ASR 最高。
- v2 更接近论文流程，但 HotFlip 后缀质量和目标函数没有处理好，导致 ASR 明显下降。
- v3 相比 v2 有恢复，尤其 Qwen2.5 Judge ASR 明显提升，但仍未超过 baseline。

因此，后续优化不应只追求“形式上更接近论文”，而应重点关注扰动是否真的转化为更具体、更任务完成的输出。下一步更有价值的方向是：限制 HotFlip token 搜索空间、对候选替换做真实前向验证、保存最佳 suffix，并把最终回答的任务完成度纳入筛选或分析。
