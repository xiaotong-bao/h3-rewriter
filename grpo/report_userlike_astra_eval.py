"""Publish a full local report immediately after each successful evaluation."""
import fcntl
import json
import os
from pathlib import Path
import time

ROOT = Path('/data/xiaotong/h3_rewriter_sft_20261002/grpo_9bv2_userlike_1k_20261007/astra_eval_every32')


def reports():
    previous = None
    index = []
    for folder in sorted(ROOT.glob('step-*')):
        if not (folder / 'COMPLETE').exists():
            continue
        step = int(folder.name.split('-')[1])
        summary = json.loads((folder / 'summary.json').read_text())
        score = next(iter(summary['models'].values()))
        assert not summary['failures'] and score['valid'] == score['expected'] == 101
        audits = [json.loads(line) for line in (folder / 'audits.jsonl').read_text().splitlines()]
        assert len(audits) == len({a['id'] for a in audits}) == 101
        severe = {a['id'] for a in audits if a['maximum_severity'] == 'severe'}
        text = [f'# GRPO step {step}：101 条 Astra 评测', '',
            f"内容分 **{score['content_score']:.2f}**；格式分 **{score['format_score']:.2f}**。",
            f"严重 {score['severe']}，一般 {score['general']}，无问题 {score['none']}；101/101 成功，最终无评分失败。", '',
            'GPT-6 Astra low；retention-single-pass-v3-pilot；HF 模板、未合并 LoRA、BF16 autocast、greedy。', '']
        if previous:
            old_step, old_score, old_severe = previous
            text += [f"相较 step {old_step}：内容分 {score['content_score']-old_score['content_score']:+.2f}；严重 case 数 {score['severe']-old_score['severe']:+d}。", '',
                '上一轮严重、本轮不再严重：' + (', '.join(sorted(old_severe-severe)) or '无') + '。',
                '本轮新增严重：' + (', '.join(sorted(severe-old_severe)) or '无') + '。', '']
        for level, title in [('severe', '严重问题'), ('general', '一般问题')]:
            text += [f'## {title}', '']
            for audit in sorted(audits, key=lambda a: a['id']):
                if audit['maximum_severity'] != level:
                    continue
                text += [f"### {audit['id']}", '', f"原始 prompt：{audit['original']}", '']
                for issue in audit['issues']:
                    text += [f"- **{issue['severity']} / {issue['category']}**：{issue['reason']}",
                        f"  原文证据：{issue['source_quote']}",
                        f"  改写证据：{issue['rewrite_quote'] or '遗漏，无对应文本'}", '']
        target = folder / 'REPORT.md'
        content = '\n'.join(text) + '\n'
        if not target.exists() or target.read_text() != content:
            temporary = folder / 'REPORT.tmp'
            temporary.write_text(content)
            temporary.replace(target)
            print('REPORT_READY', step, str(target), flush=True)
        index.append(f"- [Step {step}](step-{step:04d}/REPORT.md)：{score['content_score']:.2f} 分，{score['severe']} 个严重 case。")
        previous = step, score, severe
    (ROOT / 'REPORTS.md').write_text('# 已完成评测报告\n\n' + '\n'.join(index) + '\n')


def main():
    lock = (ROOT / 'reporter.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (ROOT / 'reporter.pid').write_text(str(os.getpid()) + '\n')
    while True:
        reports()
        if (ROOT / 'COMPLETE').exists():
            return
        time.sleep(10)


if __name__ == '__main__':
    main()
