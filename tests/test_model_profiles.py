"""Regression tests for the opt-in MOSS and MiniMax H3 profiles."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import safetensors
import safetensors.torch
import torch
import forge_neo_converter_core as core
from test_w6a8 import FakeW6A8Layout, CONFIG
from test_converter import DummyInt8Layout, DummyW4A8Layout


MOSS = [f'layers.0.proj.{name}' for name in ('qkv', 'o', 'ffn_up_gate', 'ffn_down')] + [
    f'model.diffusion_model.layers.1.proj.{name}' for name in ('q', 'k', 'v')]
H3 = [f'diffusion_model.blocks.0.{name}' for name in ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2')]


def fixture(profile):
    eligible = MOSS if profile == 'MOSS' else H3
    names = (['text_embedder.proj.weight', 'video_embedder.weight', 'shared_block_modulation.weight',
              'video_head_modulation.weight', 'final_linear_video.weight', 'layers.0.norm.weight',
              'rope.bands', 'layers.0.proj.unknown.weight'] if profile == 'MOSS' else
             ['adaln_t_table', 'blocks.0.adaln_proj.weight', 'time_embedder.proj_in.weight',
              'token_refiner.blocks.0.attn.qkv_proj.weight', 'condition_proj.weight',
              'patch_embed.weight', 'final_layer.linear.weight', 'blocks.0.norm1.weight',
              'blocks.0.attn.comfy_attention.config', 'blocks.0.attn.other.weight'])
    kept = {name: torch.full((64, 256), 1.0001, dtype=torch.float32) for name in names}
    if profile == 'MiniMax H3':
        kept['blocks.0.attn.comfy_attention.config'] = torch.tensor([123, 125], dtype=torch.uint8)
    kept['model.diffusion_model.layers.99.proj.qkv.weight' if profile == 'MOSS' else
         'model.diffusion_model.blocks.99.mlp.fc1.weight'] = torch.ones((32, 256), dtype=torch.bfloat16)
    weights = {name + '.weight': torch.ones((64, 256), dtype=torch.bfloat16) for name in eligible}
    weights.update(kept)
    return eligible, kept, weights


class ModelProfileTests(unittest.TestCase):
    def test_new_profiles_export_only_known_linears_and_preserve_source_dtype(self):
        for profile in ('MOSS', 'MiniMax H3'):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmp:
                eligible, kept, weights = fixture(profile)
                src = Path(tmp) / 'source-bf16.safetensors'
                safetensors.torch.save_file(weights, str(src), metadata={'fixture': profile, 'config': 'original'})
                with (mock.patch.object(core, 'ck', object()),
                      mock.patch.object(core, 'AsymW4A8Int8Layout', FakeW6A8Layout),
                      mock.patch.object(core, '_probe_w6a8')):
                    _, out = core.convert_model(str(src), profile, 'w6a8_convrot', 'cpu')
                with safetensors.safe_open(out, framework='pt') as h:
                    self.assertEqual(json.loads(h.metadata()['_quantization_metadata'])['layers'],
                                     {name: CONFIG for name in eligible})
                    self.assertEqual(h.metadata()['config'], 'original')
                    for name in eligible:
                        self.assertEqual(h.get_tensor(name + '.weight').shape, (64, 192))
                        self.assertEqual(h.get_tensor(name + '.weight').dtype, torch.int8)
                    for name, expected in kept.items():
                        actual = h.get_tensor(name)
                        self.assertEqual(actual.dtype, expected.dtype, name)
                        self.assertTrue(torch.equal(actual, expected), name)

    def test_int8_and_w4a8_use_same_profile_policy(self):
        for profile in ('MOSS', 'MiniMax H3'):
            for target, fmt in [('int8_convrot', 'int8_tensorwise'), ('w4a8_convrot', 'asym_w4a8_int8')]:
                with self.subTest(profile=profile, target=target), tempfile.TemporaryDirectory() as tmp:
                    eligible, kept, weights = fixture(profile)
                    # N=32 may be eligible for INT8; leave only protected tensors in this case.
                    weights.pop(next(reversed(kept)))
                    kept.pop(next(reversed(kept)))
                    src = Path(tmp) / 'source-bf16.safetensors'
                    safetensors.torch.save_file(weights, str(src))
                    with (mock.patch.object(core, 'ck', object()),
                          mock.patch.object(core, 'TensorWiseINT8Layout', DummyInt8Layout),
                          mock.patch.object(core, 'AsymW4A8Int8Layout', DummyW4A8Layout)):
                        _, out = core.convert_model(str(src), profile, target, 'cpu')
                    with safetensors.safe_open(out, framework='pt') as h:
                        layers = json.loads(h.metadata()['_quantization_metadata'])['layers']
                        self.assertEqual(set(layers), set(eligible))
                        self.assertTrue(all(conf['format'] == fmt for conf in layers.values()))
                        for name, expected in kept.items():
                            self.assertEqual(h.get_tensor(name).dtype, expected.dtype)
                            self.assertTrue(torch.equal(h.get_tensor(name), expected))

    def test_wrong_profile_does_not_replace_existing_output(self):
        for profile in ('MOSS', 'MiniMax H3'):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / 'source-bf16.safetensors'
                out = Path(tmp) / 'source-w6a8_convrot.safetensors'
                out.write_bytes(b'existing')
                safetensors.torch.save_file({'double_blocks.0.img_attn.proj.weight': torch.ones((64, 256))}, str(src))
                with (mock.patch.object(core, 'ck', object()),
                      mock.patch.object(core, 'AsymW4A8Int8Layout', FakeW6A8Layout),
                      mock.patch.object(core, '_probe_w6a8')):
                    with self.assertRaisesRegex(ValueError, 'No recognized diffusion block weights'):
                        core.convert_model(str(src), profile, 'w6a8_convrot', 'cpu')
                self.assertEqual(out.read_bytes(), b'existing')

    def test_existing_profiles_keep_legacy_dtype_policy(self):
        cfg = core.load_model_configs()['models']
        for name, profile in cfg.items():
            if name not in ('MOSS', 'MiniMax H3'):
                self.assertNotIn('quantize_key_regex', profile)
                self.assertNotIn('preserve_source_dtype', profile)
        self.assertEqual(core.preserve_tensor(torch.ones(1, dtype=torch.float32), 'model')[0].dtype, torch.float16)


if __name__ == '__main__':
    unittest.main()
