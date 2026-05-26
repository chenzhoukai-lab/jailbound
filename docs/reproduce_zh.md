# JailBound 本地复现说明

本工程按论文实现两段式流程：

1. **Safety Boundary Probing**：在 MM-SafetyBench 的原始不安全 prompt 和一个安全描述 prompt 上提取 Qwen2.5-VL 融合后的 decoder hidden states，并按层训练二分类 logistic boundary probe。
2. **Safety Boundary Crossing**：读取 probe 的法向量和边界距离，在目标层上优化视觉扰动，并从候选后缀里选择边界损失最低的文本扰动，最后调用本地 Qwen2.5-VL 生成输出。
3. **Qwen3Guard 评测**：用本地 Qwen3Guard 判断输出 `Safe/Unsafe/Controversial`，其中 `Unsafe` 计为 ASR 成功；同时用拒答模板过滤得到 `attack_effective`。

## 需要先改的路径

编辑 `configs/qwen25vl_local.json`：

- `dataset_root`：本地 MM-SafetyBench 根目录；支持根目录下直接是类别目录，也支持 `dataset_root/mm-safetybench` 或 `dataset_root/safebench`。
- `target_model_path`：本地 `Qwen2.5-VL-7B-Instruct`。
- `guard_model_path`：本地 `Qwen3Guard`。
- `image_format`：默认 `auto`，会优先找 `images`，再找 `images_figstep/images_qr/images_wr/...`。
- `attn_implementation`：默认 `sdpa`。如果遇到 `flash_attn_2_cuda ... undefined symbol`，不要用 `flash_attention_2`；等 flash-attn 和 PyTorch 版本匹配后再切回去。

## 运行

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m jailbound run --config configs/qwen25vl_local.json --limit 20
```

## 两张 H100 并行运行

推荐用 `accelerate launch` 启动两个进程：

```powershell
$env:PYTHONPATH = "$PWD/src"
accelerate launch --num_processes 2 --mixed_precision bf16 -m jailbound run --config configs/qwen25vl_local.json
```

或者直接用脚本：

```powershell
.\scripts\run_jailbound_qwen_2h100.ps1 --limit 20
```

并行逻辑：

- `probe`：每张卡提取自己分片的 hidden states，写入 `_probe_shards/rank_*.npz`，主进程合并后训练 boundary probes。
- `attack`：每张卡只优化 `rank::world_size` 的样本，写入 `_attack_shards/rank_*.jsonl`，主进程合并成 `attack_results.jsonl`。
- `eval`：每张卡评测自己的输出分片，写入 `_guard_shards/rank_*.jsonl`，主进程合并并生成 `summary.json`。

分阶段运行：

```powershell
accelerate launch --num_processes 2 --mixed_precision bf16 -m jailbound probe --config configs/qwen25vl_local.json --limit 100
accelerate launch --num_processes 2 --mixed_precision bf16 -m jailbound attack --config configs/qwen25vl_local.json --limit 20
accelerate launch --num_processes 2 --mixed_precision bf16 -m jailbound eval --config configs/qwen25vl_local.json
```

输出在 `outputs/qwen25vl_jailbound/`：

- `boundary_probes.pt`
- `attack_results.jsonl`
- `guard_eval.jsonl`
- `summary.json`
