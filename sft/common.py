"""Use the unmodified Qwen processor/template for both training and generation."""
import copy
import hashlib
import importlib.metadata
import json
import os
import pathlib

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoProcessor
from transformers.utils import is_torchcodec_available

ROOT = pathlib.Path('/work')
JOB = ROOT / 'trl_sft_official_v2_20261006'
BASE = ROOT / 'models/Qwen3.5-9B'


def processor():
    assert is_torchcodec_available() and importlib.metadata.version('torchcodec') == '0.7.0'
    p = AutoProcessor.from_pretrained(BASE)
    official = (BASE / 'chat_template.jinja').read_text()
    assert p.chat_template == official, 'Do not replace the official model template'
    p.image_processor.size = {'shortest_edge': 3136, 'longest_edge': 200704}
    p.video_processor.size = {'shortest_edge': 3136, 'longest_edge': 50176}
    p.video_processor.fps = 1.0
    p.video_processor.min_frames = 2
    p.video_processor.max_frames = 8
    p.tokenizer.padding_side = 'right'
    assert p.tokenizer.eos_token == '<|im_end|>'
    return p


def encode(p, messages, generation=False):
    return p.apply_chat_template(copy.deepcopy(messages), tokenize=True,
        add_generation_prompt=generation, enable_thinking=False, return_dict=True,
        return_tensors='pt', processor_kwargs={'fps': 1.0,
        'do_sample_frames': True, 'truncation': False})


class OfficialCollator:
    """TRL callback for mixed text/image/video. Template is never edited.

    Full official tokenization is used once. Mask the official prompt, including
    its empty think prefix; train on the unchanged teacher answer and stop token.
    Exact suffix equality prevents accidental BPE boundary/masking mistakes.
    """
    def __init__(self, p, cache=False):
        self.p = p
        self.cache = cache
        self.cache_dir = JOB / 'processed_cache'
        if cache:
            self.cache_dir.mkdir(exist_ok=True)

    def inspect(self, row, compare_prompt=False):
        prompt = json.loads(row['prompt_json'])
        full = prompt + [{'role': 'assistant', 'content': [{'type': 'text', 'text': row['answer']}]}]
        prefix = self.p.apply_chat_template(copy.deepcopy(prompt), tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        rendered = self.p.apply_chat_template(copy.deepcopy(full), tokenize=False,
            add_generation_prompt=False, enable_thinking=False)
        assert prefix.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
        assert rendered.startswith(prefix), row['id']
        suffix = rendered[len(prefix):]
        assert suffix == row['answer'].strip() + '<|im_end|>\n', row['id']
        out = encode(self.p, full)
        ids = out['input_ids'][0]
        tail = self.p.tokenizer.encode(suffix, add_special_tokens=False)
        assert ids[-len(tail):].tolist() == tail, row['id']
        start = len(ids) - len(tail)
        assert start > 0 and len(ids) <= 8192, (row['id'], len(ids))
        if compare_prompt:
            original_prompt = encode(self.p, prompt, generation=True)
            assert torch.equal(ids[:start], original_prompt['input_ids'][0]), row['id']
            for key in ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw']:
                if key in out:
                    assert torch.equal(out[key], original_prompt[key]), (row['id'], key)
        labels = out['input_ids'].clone()
        labels[:, :start] = -100
        assert self.p.tokenizer.eos_token_id in labels[0, start:].tolist()
        out['labels'] = labels
        out.pop('video_metadata', None)
        return out, {'id': row['id'], 'tokens': len(ids), 'prompt_tokens': start,
            'target_tokens': len(tail), 'input_sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
            'images': len(row['images']), 'videos': len(row['videos'])}

    def __call__(self, rows):
        assert len(rows) == 1, 'Validated microbatch size is one'
        row = rows[0]
        if not self.cache:
            return self.inspect(row)[0]
        # Cache is specific to immutable data + official template + visual limits.
        payload = {'row': row, 'template': self.p.chat_template,
            'image_size': self.p.image_processor.size, 'video_size': self.p.video_processor.size,
            'video_fps': 1, 'video_frames': [2, 8], 'decoder': 'torchcodec-0.7.0',
            'mask': 'official-prefix-answer-eos-v1'}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f'{key}.safetensors'
        if path.exists():
            with safe_open(path, framework='pt') as f:
                assert f.metadata()['fingerprint'] == key
            return load_file(path)
        out = self.inspect(row)[0]
        tensors = {k: v.contiguous() for k, v in out.items()}
        temporary = self.cache_dir / f'{key}.{os.getpid()}.tmp'
        save_file(tensors, temporary, metadata={'fingerprint': key, 'id': row['id']})
        temporary.replace(path)
        return tensors
