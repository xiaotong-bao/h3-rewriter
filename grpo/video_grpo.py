"""Carry video tensors through generation, KL scoring and buffered GRPO loss."""
import inspect,textwrap
import torch
import trl.trainer.grpo_trainer as runtime

def install_video_support():
    if getattr(runtime.GRPOTrainer, '_h3_video_compatible', False):
        return
    original=runtime.GRPOTrainer._full_logits_logps
    source=textwrap.dedent(inspect.getsource(original))
    signature='    image_position_ids=None,\n'
    assert source.count(signature)==1
    source=source.replace(signature,signature+'    pixel_values_videos=None,\n    video_grid_thw=None,\n    num_videos=None,\n')
    marker='        # Only add logits_to_keep if the model supports it\n'
    assert source.count(marker)==1
    source=source.replace(marker,'''        if pixel_values_videos is not None:
            assert video_grid_thw is not None and num_videos is not None
            counts = torch.as_tensor(num_videos, device=video_grid_thw.device)
            video_offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
            vstart, vend = int(video_offsets[start]), int(video_offsets[end])
            patch_sizes = video_grid_thw.prod(-1)
            patch_offsets = torch.cat([patch_sizes.new_zeros(1), patch_sizes.cumsum(0)])
            model_inputs['video_grid_thw'] = video_grid_thw[vstart:vend]
            model_inputs['pixel_values_videos'] = pixel_values_videos[int(patch_offsets[vstart]):int(patch_offsets[vend])]
''' + marker)
    namespace=dict(original.__globals__);exec(compile(source,'<video-aware-grpo-logps>','exec'),namespace)
    runtime.GRPOTrainer._full_logits_logps=namespace['_full_logits_logps']
    old_split,old_unsplit=runtime.split_pixel_values_by_grid,runtime.unsplit_pixel_values_by_grid
    def split(batch):
        batch=old_split(batch)
        if isinstance(batch.get('pixel_values_videos'),torch.Tensor):
            grids=batch['video_grid_thw'];counts=batch['num_videos'];grid_groups=list(torch.split(grids,counts))
            patches=[int(g.prod(-1).sum()) for g in grid_groups]
            batch={**batch,'video_grid_thw':grid_groups,'pixel_values_videos':list(torch.split(batch['pixel_values_videos'],patches))}
        return batch
    def unsplit(batch):
        batch=old_unsplit(batch)
        if isinstance(batch.get('pixel_values_videos'),list):
            batch={**batch,'pixel_values_videos':torch.cat(batch['pixel_values_videos']),'video_grid_thw':torch.cat(batch['video_grid_thw'])}
        return batch
    runtime.split_pixel_values_by_grid=split;runtime.unsplit_pixel_values_by_grid=unsplit
    runtime.GRPOTrainer._h3_video_compatible = True


def video_trainer_class(parent):
    class VideoTrainer(parent):
        def _tokenize_prompts(self,prompts):
            ids,images,fields=super()._tokenize_prompts(prompts)
            fields.pop('video_metadata',None)
            self._video_fields={k:fields[k] for k in ('pixel_values_videos','video_grid_thw') if k in fields}
            if self._video_fields:
                self._video_fields['num_videos']=[sum(p.get('type')=='video' for m in prompt for p in m['content'] if isinstance(p,dict)) for prompt in prompts]
            return ids,images,fields
        def _get_per_token_logps_and_entropies(self,model,input_ids,*args,**kwargs):
            video=getattr(self,'_video_fields',{})
            if video:
                for k,v in video.items():kwargs[k]=v.to(input_ids.device) if isinstance(v,torch.Tensor) else v
                assert len(kwargs['num_videos'])==len(input_ids)
                if kwargs.get('mm_token_type_ids') is None:
                    kinds=torch.zeros_like(input_ids)
                    kinds[input_ids==self._tokenizer.convert_tokens_to_ids('<|image_pad|>')]=1
                    kinds[input_ids==self._tokenizer.convert_tokens_to_ids('<|video_pad|>')]=2
                    kwargs['mm_token_type_ids']=kinds
            return super()._get_per_token_logps_and_entropies(model,input_ids,*args,**kwargs)
        def _generate_and_score_completions(self,inputs):
            output=super()._generate_and_score_completions(inputs)
            output.update(getattr(self,'_video_fields',{}))
            return output
        def compute_loss(self,model,inputs,*args,**kwargs):
            self._video_fields={k:inputs[k] for k in ('pixel_values_videos','video_grid_thw','num_videos') if k in inputs}
            return super().compute_loss(model,inputs,*args,**kwargs)
    return VideoTrainer
