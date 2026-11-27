"""W6A8 export contract and optional real CUDA integration tests."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import safetensors
import safetensors.torch
import torch
import forge_neo_converter_core as core


class FakeW6A8Layout:
    calls = []

    @classmethod
    def quantize(cls, tensor, **kwargs):
        n, k = tensor.shape
        cls.calls.append((n, k, kwargs))
        return torch.zeros((n, k * 3 // 4), dtype=torch.int8), SimpleNamespace(
            group_size=kwargs['group_size'], convrot_groupsize=kwargs['convrot_groupsize'],
            codebook=None, scale=torch.ones((n, k // 32), dtype=torch.float32),
            s_channel=torch.ones(n, dtype=torch.float32),
        )

    @staticmethod
    def state_dict_tensors(q, p):
        return {'': q, '_s_rel': p.scale, '_s_channel': p.s_channel}


CONFIG = {'format': 'w6a8_int8', 'group_size': 32, 'convrot': True, 'convrot_groupsize': 256}


def check_export(test, output, eligible, unsupported):
    with safetensors.safe_open(str(output), framework='pt') as h:
        meta = json.loads(h.metadata()['_quantization_metadata'])
        test.assertEqual(meta['layers'], {name: CONFIG for name in eligible})
        test.assertEqual(h.metadata()['fixture'], 'w6a8')
        for name, (n, k) in eligible.items():
            q = h.get_tensor(name + '.weight')
            test.assertEqual(tuple(q.shape), (n, k * 3 // 4))
            test.assertEqual(q.dtype, torch.int8)
            test.assertEqual(tuple(h.get_tensor(name + '.weight_s_rel').shape), (n, k // 32))
            test.assertEqual(tuple(h.get_tensor(name + '.weight_s_channel').shape), (n,))
            test.assertEqual(json.loads(bytes(h.get_tensor(name + '.comfy_quant').tolist())), CONFIG)
            test.assertNotIn(name + '.weight_codebook', h.keys())
        for name, tensor in unsupported.items():
            test.assertNotIn(name + '.comfy_quant', h.keys())
            test.assertTrue(torch.equal(h.get_tensor(name + '.weight'), tensor))


class W6A8Tests(unittest.TestCase):
    def test_export_rectangular_and_unsupported_shapes(self):
        eligible = {'blocks.0.linear': (128, 256), 'blocks.1.linear': (64, 512),
                    'blocks.2.linear': (256, 256), 'blocks.3.linear': (192, 256)}
        unsupported = {name: torch.ones(shape, dtype=torch.float16) for name, shape in
                       {'blocks.4.linear': (32, 256), 'blocks.5.linear': (128, 192),
                        'blocks.6.linear': (256, 128)}.items()}
        weights = {name + '.weight': torch.ones(shape, dtype=torch.float16) for name, shape in eligible.items()}
        weights.update({name + '.weight': tensor for name, tensor in unsupported.items()})
        FakeW6A8Layout.calls.clear()
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'model-fp16.safetensors'
            safetensors.torch.save_file(weights, str(source), metadata={'fixture': 'w6a8'})
            with (mock.patch.object(core, 'ck', object()),
                  mock.patch.object(core, 'AsymW4A8Int8Layout', FakeW6A8Layout),
                  mock.patch.object(core, '_probe_w6a8') as probe):
                _, output = core.convert_model(str(source), 'Other/Unknown', 'w6a8_convrot', 'cpu')
                probe.assert_called_once()
            self.assertEqual(Path(output).name, 'model-w6a8_convrot.safetensors')
            check_export(self, output, eligible, unsupported)
        self.assertCountEqual([(n,k) for n,k,_ in FakeW6A8Layout.calls], eligible.values())
        for _,_,kw in FakeW6A8Layout.calls:
            self.assertEqual(kw['bits'], 6)
            self.assertFalse(kw['codebook'])
            self.assertEqual(kw['group_size'], 32)
            self.assertEqual(kw['convrot_groupsize'], 256)

    def test_all_unsupported_writes_no_output(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'model-fp16.safetensors'
            safetensors.torch.save_file({'layer.weight': torch.ones((128, 192), dtype=torch.float16)}, str(source))
            with (mock.patch.object(core, 'ck', object()),
                  mock.patch.object(core, 'AsymW4A8Int8Layout', FakeW6A8Layout),
                  mock.patch.object(core, '_probe_w6a8')):
                with self.assertRaisesRegex(RuntimeError, 'No compatible W6A8'):
                    core.convert_model(str(source), 'Other/Unknown', 'w6a8_convrot', 'cpu')
            self.assertFalse((Path(d) / 'model-w6a8_convrot.safetensors').exists())

    def test_backend_failure_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'model-fp16.safetensors'
            target = Path(d) / 'model-w6a8_convrot.safetensors'
            safetensors.torch.save_file({'layer.weight': torch.ones((128, 256), dtype=torch.float16)}, str(source))
            target.write_bytes(b'existing output')
            with (mock.patch.object(core, 'ck', object()),
                  mock.patch.object(core, 'AsymW4A8Int8Layout', FakeW6A8Layout),
                  mock.patch.object(core, '_probe_w6a8'),
                  mock.patch.object(core, '_w6a8_quantize', side_effect=RuntimeError('kernel failure'))):
                with self.assertRaisesRegex(RuntimeError, 'W6A8 quantization failed'):
                    core.convert_model(str(source), 'Other/Unknown', 'w6a8_convrot', 'cpu')
            self.assertEqual(target.read_bytes(), b'existing output')

    @unittest.skipUnless(torch.cuda.is_available() and core.AsymW4A8Int8Layout is not None,
                         'real CUDA + comfy-kitchen required')
    def test_real_cuda_export(self):
        eligible = {'blocks.0.linear': (128, 256), 'blocks.1.linear': (64, 256)}
        unsupported = {'blocks.2.linear': torch.ones((128, 192), dtype=torch.float16)}
        gen = torch.Generator().manual_seed(1234)
        weights = {name + '.weight': (torch.randn(shape, generator=gen) * .05).half() for name, shape in eligible.items()}
        weights.update({name + '.weight': tensor for name, tensor in unsupported.items()})
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'model-fp16.safetensors'
            safetensors.torch.save_file(weights, str(source), metadata={'fixture': 'w6a8'})
            _, output = core.convert_model(str(source), 'Other/Unknown', 'w6a8_convrot', 'cuda', log=print)
            check_export(self, output, eligible, unsupported)


if __name__ == '__main__':
    unittest.main()
