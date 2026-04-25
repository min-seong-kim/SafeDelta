'''
# Author: Ning LU
# This code is adapted from https://github.com/IST-DASLab/sparsegpt
# The current implementation follows an online processing approach for better code readability.


python llama2/run_safedelta.py \
    --model_name_align kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --model_name_ft finetuned_models/gsm8k-llama2-7b-chat-safeft \
    --scale 0.5 \
    --safe_data_path ./llama2/safedelta/data/circuit_breakers_train.json \
    --upload_name kmseong/llama2-7b-chat-safedelta-scale0.5


'''

import os
import random
import warnings

try:
    import wandb

    has_wandb = True
except:
    has_wandb = False

import math
import time

import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
import json
import fire

from configs import fsdp_config, train_config

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    default_data_collator,
)

transformers.set_seed(0)

from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM, AutoTokenizer
from safedelta.safedelta_runner import get_safe_data_systemprompt, find_layers, SafeDeltaRunner, get_safe_data, get_circuit_breakers_data

DEBUG = True

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def recovery_safety(
    model_name_align: str,
    model_name_ft: str,
    s: float,
    st_layer: int = 0,
    safe_data_path: str = None,
    **kwargs,
):
    ## load model

    align_model = LlamaForCausalLM.from_pretrained(
        model_name_align,
        return_dict=True,
        device_map="cuda",
        low_cpu_mem_usage=True,
        torch_dtype="auto",
    )

    ft_model = LlamaForCausalLM.from_pretrained(
        model_name_ft,
        return_dict=True,
        # load_in_8bit=False,
        device_map="cuda",
        low_cpu_mem_usage=True,
        torch_dtype="auto",
    )

    # Load the tokenizer and add special tokens
    if 'llama3' in model_name_ft:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_align
        )
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        if 'llama2' not in model_name_ft:
            warnings.warn("Warning: Current implementation only supports LLaMA-2.", UserWarning)

        tokenizer = LlamaTokenizer.from_pretrained(model_name_align)
        tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )

    final_model = run_safedelta(
        align_model, ft_model, tokenizer, s, st_layer,
        safe_data_path=safe_data_path,
        model_name_align=model_name_align,
    )

    model_save_path = model_name_ft + f'-SafeDelta-s{s}'

    # convert
    final_model.to(torch.float16)
    tokenizer.save_pretrained(model_save_path)
    final_model.save_pretrained(model_save_path)

    print('Save to', model_save_path)

    upload_name = kwargs.pop('upload_name', None)
    hf_token = kwargs.pop('hf_token', None)
    if upload_name:
        print(f'Uploading to HuggingFace Hub: {upload_name} ...')
        final_model.push_to_hub(upload_name, token=hf_token)
        tokenizer.push_to_hub(upload_name, token=hf_token)
        print(f'Uploaded to https://huggingface.co/{upload_name}')


@torch.no_grad()
def run_safedelta(align_model, ft_model, tokenizer, s, st_layer_idx, nsamples=128,
                  safe_data_path=None, model_name_align=''):
    use_cache = align_model.config.use_cache
    align_model.config.use_cache = False

    # batch_size = 1
    seq_len = 512
    # nsamples = 128

    # dataloader = []

    # sys_prompts_list = ['pure_bad', 'aoa', 'math', 'pure_bad']
    # for idx, sys_prompt in enumerate(sys_prompts_list):
    #     cur_dataloader = get_safe_data_systemprompt(nsamples // len(sys_prompts_list), tokenizer, seq_len,
    #                                                 template=sys_prompt, seed=idx)
    #     dataloader.extend(cur_dataloader)

    if safe_data_path is not None:
        dataloader = get_circuit_breakers_data(
            nsamples, tokenizer, seq_len,
            model_name_or_path=model_name_align,
            data_path=safe_data_path,
        )
    else:
        dataloader = get_safe_data(nsamples, tokenizer, seq_len)

    align_layers = align_model.model.layers
    ft_layers = ft_model.model.layers
    dtype = next(iter(align_model.model.parameters())).dtype
    device = torch.device("cuda")

    inps = []
    # tars = []
    attention_mask = []
    position_ids = []
    position_embeddings = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            attention_mask.append(kwargs.get("attention_mask"))
            position_ids.append(kwargs.get("position_ids"))
            position_embeddings.append(kwargs.get("position_embeddings"))

            raise ValueError

    align_layers[st_layer_idx] = Catcher(align_layers[st_layer_idx])

    for batch in dataloader:
        try:
            align_model(batch[0].to(device))
        except ValueError:
            pass

    align_layers[0] = align_layers[0].module
    torch.cuda.empty_cache()

    # outs = torch.zeros_like(inps)
    outs = [None for _ in range(nsamples)]
    align_model.config.use_cache = use_cache

    # attention_mask = cache['attention_mask']
    # position_ids = cache['position_ids']

    print('Ready.')

    # TODO: adapt for higher version of transformers package
    # current, transformers <= v4.46

    inps = [inp.squeeze(0).to(device) for inp in inps]
    position_ids = [pids.to(device) if pids is not None else None for pids in position_ids]
    position_embeddings = [
        (pe[0].to(device), pe[1].to(device)) if pe is not None else None
        for pe in position_embeddings
    ]

    print('Start Online Safe Delta.')

    for i in tqdm(range(st_layer_idx, len(align_layers))):
        align_layer = align_layers[i]
        ft_layer = ft_layers[i]
        align_subset = find_layers(align_layer)
        ft_subset = find_layers(ft_layer)

        gpts = {}
        for name in align_subset:
            gpts[name] = SafeDeltaRunner(align_subset[name], ft_subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)  # tar

            return tmp

        handles = []
        for name in gpts:
            handles.append(align_subset[name].register_forward_hook(add_batch(name)))

        for j in range(nsamples):
            layer_kwargs = dict(
                attention_mask=attention_mask[j],
                position_ids=position_ids[j],
            )
            if position_embeddings[j] is not None:
                layer_kwargs["position_embeddings"] = position_embeddings[j]
            outs[j] = align_layer(
                inps[j].unsqueeze(0),
                **layer_kwargs,
            )[0].squeeze(0)

        for h in handles:
            h.remove()

        for name in gpts:
            gpts[name].adjust_delta(
                s,
                percdamp=0.01,
                blocksize=2048,
            )
            gpts[name].free()

        align_layers[i] = align_layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    return align_model


def main(model_name_align: str = 'ckpts/llama2-7b-chat-hf',
         model_name_ft: str = 'finetuned_models/purebad100-7b-full',
         scale: float = 0.1,
         st_layer: int = 0,
         upload_name: str = None,
         hf_token: str = None,
         **kwargs):
    recovery_safety(model_name_align, model_name_ft, scale, st_layer,
                    upload_name=upload_name, hf_token=hf_token, **kwargs)


if __name__ == "__main__":
    fire.Fire(main)
