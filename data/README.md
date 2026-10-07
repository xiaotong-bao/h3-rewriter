# 数据准备

实际原始数据已完成 9899 条 teacher 标签，固定排除 40 条后为 9659 train / 200 val。
生产 split 和数据 S3 地址见 [SFT runbook](../docs/runbooks/sft.md#当前任务和数据)。
这里提供已有标签之后的 split 和 chat 导出，不包含原始 Context-IR API 标注驱动。
原始10k配额计划留在历史归档；当前数据以SFT runbook为准。

## 固定 split

`split_dataset.py` 从成功的原始标签与请求 manifest 制作数据。保留 teacher target，按原文/媒体 leakage keys 排除冻结验证集重合项。重现生产 split 应恢复原冻结 val，不能重新随机生成另一份验证集。输出目录必须是新的空目录。

```bash
python data/split_dataset.py \
  --source /work/source/h3_rewrite_results.jsonl \
  --manifest /work/source/manifest_api.jsonl \
  --frozen-val /work/frozen_val.jsonl \
  --out /work/dataset
```

`split_dataset.py` 的 system 是历史数据准备 system。当前训练通过 `sft/prepare.py` 替换为 `system_prompt_v2.txt` 原始字节，不把历史 system 当作 retention v2。

## 原始 chat 导出

在实验根挂载为 `/work` 的容器中：

```bash
python /path/to/h3-rewriter/data/export_chat.py
```

读取 `/work/source/manifest_api.jsonl`、`/work/dataset/{train_partial,val}.jsonl` 和原始 system，写 `/work/lf_dataset/`。文件名中的 `lf` 是已有 ShareGPT 导出布局；当前训练使用 LLaMA-Factory；TRL导出用于历史实验。原始 user text、teacher answer、任务/时长/比例及引用顺序保持不变。
媒体需已在 `/work/media/`；文件名由源 media_path 字符串 SHA256 + 扩展名生成，不代表文件内容 SHA256。
音频 waveform 不送给 vision-only Qwen；保留用户请求和音频关系，明确不猜测未提供内容。
历史TRL实验随后执行 `sft/prepare.py`；当前LF任务使用冻结数据快照，见SFT runbook。
