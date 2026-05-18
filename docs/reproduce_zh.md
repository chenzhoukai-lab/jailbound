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

## 运行

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m jailbound run --config configs/qwen25vl_local.json --limit 20
```

分阶段运行：

```powershell
python -m jailbound probe --config configs/qwen25vl_local.json --limit 100
python -m jailbound attack --config configs/qwen25vl_local.json --limit 20
python -m jailbound eval --config configs/qwen25vl_local.json
```

输出在 `outputs/qwen25vl_jailbound/`：

- `boundary_probes.pt`
- `attack_results.jsonl`
- `guard_eval.jsonl`
- `summary.json`

