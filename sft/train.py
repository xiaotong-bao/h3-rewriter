import argparse
import hashlib
import json
import os
import pathlib
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import Qwen3_5ForConditionalGeneration, TrainerCallback
from trl import SFTConfig, SFTTrainer

from common import BASE, JOB, OfficialCollator, processor
from epoch_benchmark import EpochBenchmark


def load(split):
    return [json.loads(line) for line in (JOB / f'{split}.jsonl').open()]


class Progress(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            target = pathlib.Path(args.output_dir)
            target.mkdir(parents=True, exist_ok=True)
            source_hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                for f in JOB.glob('*.py')}
            (target / 'training_manifest.json').write_text(json.dumps({
                'base': str(BASE), 'fresh_sft': True, 'epochs': args.num_train_epochs,
                'max_steps': state.max_steps, 'world_size': args.world_size,
                'effective_batch_size': args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size,
                'training_args': args.to_dict(), 'source_sha256': source_hashes,
                'data_report': json.loads((JOB / 'data_report.json').read_text()),
                'official_template_receipt': json.loads((JOB / 'official_template_receipt.json').read_text())}, indent=2))

    def on_epoch_end(self, args, state, control, **kwargs):
        if not args.output_dir.endswith('/smoke'):
            control.should_save = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            (JOB / 'progress.json').write_text(json.dumps({'step': state.global_step,
                'max_steps': state.max_steps, 'epoch': state.epoch, 'time': time.time(),
                'output_dir': args.output_dir, 'metrics': logs}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--resume')
    a = parser.parse_args()
    torch.set_num_threads(1)
    p = processor()
    template_sha = hashlib.sha256(p.chat_template.encode()).hexdigest()
    if not a.smoke:
        gate = json.loads((JOB / 'preflight_report.json').read_text())
        assert gate['passed'] and gate['rows'] == 9859 and gate['template_sha256'] == template_sha
    rows, validation = load('train'), load('val')
    if a.smoke:
        selected = {}
        for row in rows:
            kind = ('video' if row['videos'] else 'image' if row['images'] else 'text')
            selected.setdefault(kind, row)
        rows = list(selected.values())
        assert len(rows) == 3
    model = Qwen3_5ForConditionalGeneration.from_pretrained(BASE,
        dtype=torch.bfloat16, attn_implementation='sdpa')
    model.generation_config.eos_token_id = p.tokenizer.eos_token_id
    model.generation_config.pad_token_id = p.tokenizer.pad_token_id
    model.config.use_cache = False
    # Language-only LoRA; keep vision encoder and multimodal projector frozen.
    targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',
               'in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj']
    modules = [name for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and name.startswith('model.language_model.')
               and name.rsplit('.', 1)[-1] in targets]
    assert modules and not any('visual' in name for name in modules)
    model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=.05,
        target_modules=modules, task_type='CAUSAL_LM'))
    model.print_trainable_parameters()
    assert all('language_model' in name for name, param in model.named_parameters() if param.requires_grad)
    output = JOB / ('smoke' if a.smoke else 'run')
    args = SFTConfig(output_dir=str(output), num_train_epochs=a.epochs,
        max_steps=3 if a.smoke else -1, learning_rate=1e-4,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if a.smoke else 4,
        lr_scheduler_type='cosine', warmup_steps=0 if a.smoke else 18,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        max_length=None, packing=False, assistant_only_loss=False,
        completion_only_loss=False, dataset_kwargs={'skip_prepare_dataset': True},
        remove_unused_columns=False, ddp_find_unused_parameters=False,
        logging_steps=1, save_strategy='no' if a.smoke else 'steps', save_steps=25,
        eval_strategy='no' if a.smoke else 'epoch',
        save_total_limit=None, report_to='none', dataloader_num_workers=0,
        seed=42, data_seed=42, loss_type='nll')
    trainer = SFTTrainer(model=model, args=args, processing_class=p,
        train_dataset=Dataset.from_list(rows), eval_dataset=Dataset.from_list(validation),
        data_collator=OfficialCollator(p, cache=True),
        callbacks=[Progress(), *([] if a.smoke else [EpochBenchmark(p)])])
    assert trainer.processing_class.chat_template == p.chat_template
    trainer.train(resume_from_checkpoint=a.resume)
    trainer.save_model(str(output / 'final_adapter'))
    if trainer.is_world_process_zero():
        p.save_pretrained(output / 'final_adapter')
        model.generation_config.save_pretrained(output / 'final_adapter')
        (output / 'COMPLETE').write_text('complete\n')


if __name__ == '__main__':
    main()
