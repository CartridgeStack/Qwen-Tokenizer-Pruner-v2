import os
import json
import shutil
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Original full-model functions (kept for reference / small models)
# ─────────────────────────────────────────────────────────────────────────────

def saving_updated_qwenvl(old_model, new_vocab_size, token_mapping, output_path):
    new_embeds = torch.nn.Embedding(new_vocab_size, old_model.config.hidden_size, dtype=old_model.transformer.wte.weight.dtype)
    new_lm_head = torch.nn.Linear(old_model.config.hidden_size, new_vocab_size, bias=False, dtype=old_model.lm_head.weight.dtype)
    assert len(set(token_mapping)) == new_vocab_size
    new_embeds.weight.data = old_model.transformer.wte.weight.data[torch.LongTensor(token_mapping, device=old_model.device)]
    new_lm_head.weight.data = old_model.lm_head.weight.data[torch.LongTensor(token_mapping, device=old_model.device)]
    old_model.transformer.wte.weight = new_embeds.weight
    old_model.lm_head.weight = new_lm_head.weight
    old_model.transformer.wte.num_embeddings = new_vocab_size
    old_model.lm_head.out_features = new_vocab_size
    old_model.config.__dict__['vocab_size'] = new_vocab_size
    old_model.config.__dict__['_name_or_path'] = output_path
    old_model.config.__dict__['visual']["image_start_id"] = token_mapping.index(old_model.config.__dict__['visual']["image_start_id"])
    old_model.generation_config.__dict__['eos_token_id'] = token_mapping.index(old_model.generation_config.__dict__['eos_token_id'])
    old_model.generation_config.__dict__['pad_token_id'] = token_mapping.index(old_model.generation_config.__dict__['pad_token_id'])
    print(f"Saving new model ckpt to {output_path}")
    old_model.save_pretrained(output_path)


def saving_updated_qwen(old_model, new_vocab_size, token_mapping, output_path):
    new_embeds = torch.nn.Embedding(new_vocab_size, old_model.config.hidden_size, dtype=old_model.transformer.wte.weight.dtype)
    new_lm_head = torch.nn.Linear(old_model.config.hidden_size, new_vocab_size, bias=False, dtype=old_model.lm_head.weight.dtype)
    assert len(set(token_mapping)) == new_vocab_size
    new_embeds.weight.data = old_model.transformer.wte.weight.data[torch.LongTensor(token_mapping, device=old_model.device)]
    new_lm_head.weight.data = old_model.lm_head.weight.data[torch.LongTensor(token_mapping, device=old_model.device)]
    old_model.transformer.wte.weight = new_embeds.weight
    old_model.lm_head.weight = new_lm_head.weight
    old_model.transformer.wte.num_embeddings = new_vocab_size
    old_model.lm_head.out_features = new_vocab_size
    old_model.config.__dict__['vocab_size'] = new_vocab_size
    old_model.config.__dict__['_name_or_path'] = output_path
    old_model.generation_config.__dict__['eos_token_id'] = token_mapping.index(old_model.generation_config.__dict__['eos_token_id'])
    old_model.generation_config.__dict__['pad_token_id'] = token_mapping.index(old_model.generation_config.__dict__['pad_token_id'])
    print(f"Saving new model ckpt to {output_path}")
    old_model.save_pretrained(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Low-memory shard-by-shard implementation
#
# Strategy: never load the full model. Stream each safetensors shard, slice
# any tensor whose first dimension equals old_vocab_size, and write it back.
# Peak RAM ≈ one shard (~4 GB) + tokenizer.
# ─────────────────────────────────────────────────────────────────────────────

def _remap_token_id(token_id, mapping_new2old):
    if token_id is None:
        return None
    try:
        return mapping_new2old.index(token_id)
    except ValueError:
        return token_id


def _remap_token_id_field(value, mapping_new2old):
    if isinstance(value, list):
        return [_remap_token_id(v, mapping_new2old) for v in value]
    return _remap_token_id(value, mapping_new2old)


def saving_updated_qwen_lowmem(
    old_model_path,
    new_vocab_size,
    mapping_new2old,
    output_path,
    old_vocab_size,
    config=None,          # kept for API compatibility, no longer required
):
    """
    Slice all vocab-sized weight tensors in place, one safetensors shard at a
    time.  Any 2-D tensor whose shape[0] == old_vocab_size is treated as an
    embedding / lm-head and re-indexed via mapping_new2old.

    Works for Qwen 1.x, Qwen2, Qwen3, Qwen3.5 multimodal, and similar HF
    models — no hard-coded key names required.
    """
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("Please install safetensors: pip install safetensors")

    mapping_tensor = torch.LongTensor(mapping_new2old)

    # ── 1. Copy all non-safetensors files (configs, tokenizer files, etc.) ───
    print("==> Copying config and tokenizer files...")
    SKIP_EXTENSIONS = {'.safetensors'}
    SKIP_NAMES      = {'model.safetensors.index.json'}
    for fname in os.listdir(old_model_path):
        src = os.path.join(old_model_path, fname)
        dst = os.path.join(output_path, fname)
        if not os.path.isfile(src):
            continue
        ext = os.path.splitext(fname)[1]
        if ext in SKIP_EXTENSIONS or fname in SKIP_NAMES:
            continue
        shutil.copy2(src, dst)
        print(f"  Copied: {fname}")

    # ── 2. Patch config.json ──────────────────────────────────────────────────
    config_path = os.path.join(output_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        patched = []
        if 'vocab_size' in cfg:
            cfg['vocab_size'] = new_vocab_size
            patched.append('vocab_size')
        if isinstance(cfg.get('text_config'), dict) and 'vocab_size' in cfg['text_config']:
            cfg['text_config']['vocab_size'] = new_vocab_size
            patched.append('text_config.vocab_size')
        cfg['_name_or_path'] = output_path
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"==> config.json patched: {', '.join(patched)} => {new_vocab_size}")

    # ── 3. Patch generation_config.json ──────────────────────────────────────
    gen_cfg_path = os.path.join(output_path, 'generation_config.json')
    if os.path.exists(gen_cfg_path):
        with open(gen_cfg_path, 'r', encoding='utf-8') as f:
            gen_cfg = json.load(f)
        for field in ('eos_token_id', 'pad_token_id', 'bos_token_id'):
            if field in gen_cfg and gen_cfg[field] is not None:
                old_val = gen_cfg[field]
                gen_cfg[field] = _remap_token_id_field(old_val, mapping_new2old)
                print(f"  generation_config.json: {field} {old_val} => {gen_cfg[field]}")
        with open(gen_cfg_path, 'w', encoding='utf-8') as f:
            json.dump(gen_cfg, f, indent=2, ensure_ascii=False)

    # ── 4. Locate shards ──────────────────────────────────────────────────────
    index_path      = os.path.join(old_model_path, 'model.safetensors.index.json')
    single_shard    = os.path.join(old_model_path, 'model.safetensors')

    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)
        weight_map  = index['weight_map']
        shards      = sorted(set(weight_map.values()))
        new_weight_map = {}
        multi_shard = True
    elif os.path.exists(single_shard):
        shards      = ['model.safetensors']
        weight_map  = {}
        new_weight_map = {}
        multi_shard = False
    else:
        raise FileNotFoundError(
            f"No safetensors files found in {old_model_path}."
        )

    # ── 5. Process each shard ─────────────────────────────────────────────────
    vocab_tensors_patched = []

    for shard_name in shards:
        shard_src = os.path.join(old_model_path, shard_name)
        shard_dst = os.path.join(output_path, shard_name)
        print(f"==> Processing shard: {shard_name}")

        tensors = {}
        with safe_open(shard_src, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                # Auto-detect any vocab-row tensor by shape
                if len(tensor.shape) == 2 and tensor.shape[0] == old_vocab_size:
                    print(f"  Slicing {key}: [{old_vocab_size}, {tensor.shape[1]}]"
                          f" -> [{new_vocab_size}, {tensor.shape[1]}]")
                    tensors[key] = tensor[mapping_tensor].contiguous()
                    vocab_tensors_patched.append(key)
                else:
                    tensors[key] = tensor

                if multi_shard:
                    new_weight_map[key] = shard_name

        save_file(tensors, shard_dst)
        print(f"  Saved: {shard_dst}")
        del tensors   # free RAM before next shard

    if not vocab_tensors_patched:
        print("WARNING: No vocab-sized tensors found. "
              "Check that old_vocab_size is correct.")
    else:
        print(f"==> Patched vocab tensors: {vocab_tensors_patched}")

    # ── 6. Write updated shard index ─────────────────────────────────────────
    if multi_shard:
        new_index = {
            'metadata':   index.get('metadata', {}),
            'weight_map': new_weight_map,
        }
        new_index_path = os.path.join(output_path, 'model.safetensors.index.json')
        with open(new_index_path, 'w', encoding='utf-8') as f:
            json.dump(new_index, f, indent=2)
        print(f"==> Saved updated shard index: {new_index_path}")

    print(f"==> Model weights done. Output: {output_path}")
