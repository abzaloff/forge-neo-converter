import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr
import safetensors
import safetensors.torch
import torch

import forge_neo_converter_core as core


ROOT = Path(__file__).resolve().parents[1]
UI_SCRIPT = ROOT / "scripts" / "forge_neo_converter.py"


class DummyInt8Layout:
    @staticmethod
    def quantize(tensor, **_kwargs):
        return tensor.to(torch.int8), torch.ones(1, dtype=torch.float32, device=tensor.device)

    @staticmethod
    def state_dict_tensors(qdata, params):
        return {"": qdata, "_scale": params}


class DummyW4A8Layout:
    quantize_kwargs = None

    @classmethod
    def quantize(cls, tensor, **kwargs):
        cls.quantize_kwargs = kwargs
        qdata = tensor[:, ::2].to(torch.int8)
        params = {
            "s_rel": torch.ones(
                tensor.size(0),
                tensor.size(1) // kwargs["group_size"],
                dtype=torch.float32,
                device=tensor.device,
            ),
            "s_channel": torch.ones(tensor.size(0), dtype=torch.float32, device=tensor.device),
        }
        return qdata, params

    @staticmethod
    def state_dict_tensors(qdata, params):
        return {"": qdata, "_s_rel": params["s_rel"], "_s_channel": params["s_channel"]}


class CoreTests(unittest.TestCase):
    def test_text_encoder_weights_are_not_blocked_by_text_prefix(self):
        tensor = torch.zeros((16, 16), dtype=torch.float16)
        key = "text_model.encoder.layers.0.mlp.fc1.weight"

        self.assertFalse(core.can_quantize_weight(key, tensor))
        self.assertTrue(core.can_quantize_weight(key, tensor, protected_substrings=()))

    def test_high_precision_streaming_does_not_load_full_state_dict(self):
        state_dict = {
            "layer.weight": torch.ones((16, 16), dtype=torch.float16),
            "layer.bias": torch.ones(16, dtype=torch.float16),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "model-bf16.safetensors")
            safetensors.torch.save_file(state_dict, source)

            with mock.patch.object(
                core.safetensors.torch,
                "load_file",
                side_effect=AssertionError("full state dict load is not allowed"),
            ):
                _, output = core.convert_model(
                    source, "Other/Unknown", "fp16", "cpu"
                )

            converted = safetensors.torch.load_file(output)
            self.assertEqual(set(converted), set(state_dict))

    def test_text_encoder_quantization_uses_profile_and_preserves_embeddings_in_bf16(self):
        state_dict = {
            "text_model.encoder.layers.0.mlp.fc1.weight": torch.ones((16, 16), dtype=torch.float32),
            "text_model.embeddings.token_embedding.weight": torch.ones((16, 16), dtype=torch.float32),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "encoder-bf16.safetensors")
            safetensors.torch.save_file(state_dict, source)

            with (
                mock.patch.object(core, "ck", object()),
                mock.patch.object(core, "TensorWiseINT8Layout", DummyInt8Layout),
            ):
                _, output = core.convert_text_encoder(source, "int8", "cpu")

            converted = safetensors.torch.load_file(output)
            self.assertIn("text_model.encoder.layers.0.mlp.fc1.comfy_quant", converted)
            self.assertEqual(
                converted["text_model.embeddings.token_embedding.weight"].dtype,
                torch.bfloat16,
            )

            with safetensors.safe_open(output, framework="pt") as handle:
                metadata = handle.metadata()
            quantization = json.loads(metadata["_quantization_metadata"])
            self.assertIn("text_model.encoder.layers.0.mlp.fc1", quantization["layers"])

    def test_w4a8_convrot_uses_asym_layout_and_forge_metadata(self):
        state_dict = {
            "model.diffusion_model.blocks.1.attn.proj.weight": torch.ones(
                (256, 256), dtype=torch.float16
            ),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "model-fp16.safetensors")
            safetensors.torch.save_file(state_dict, source)
            logs = []

            with (
                mock.patch.object(core, "ck", object()),
                mock.patch.object(core, "AsymW4A8Int8Layout", DummyW4A8Layout),
            ):
                _, output = core.convert_model(
                    source,
                    "Other/Unknown",
                    "w4a8_convrot",
                    "cpu",
                    log=logs.append,
                )

            self.assertEqual(os.path.basename(output), "model-w4a8_convrot.safetensors")
            converted = safetensors.torch.load_file(output)
            layer = "model.diffusion_model.blocks.1.attn.proj"
            self.assertIn(f"{layer}.weight_s_rel", converted, logs)
            self.assertIn(f"{layer}.weight_s_channel", converted)
            self.assertIn(f"{layer}.comfy_quant", converted)
            self.assertEqual(DummyW4A8Layout.quantize_kwargs["group_size"], 16)
            self.assertEqual(DummyW4A8Layout.quantize_kwargs["convrot_groupsize"], 256)
            self.assertEqual(
                DummyW4A8Layout.quantize_kwargs["scale_dtype"],
                torch.float8_e4m3fn,
            )

            with safetensors.safe_open(output, framework="pt") as handle:
                metadata = handle.metadata()
            quantization = json.loads(metadata["_quantization_metadata"])
            self.assertEqual(
                quantization["layers"][layer],
                {
                    "format": "asym_w4a8_int8",
                    "group_size": 16,
                    "convrot_groupsize": 256,
                },
            )

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "PyTorch has no FP8 dtype")
    def test_streaming_fp8_input_applies_weight_scale_before_fp16_conversion(self):
        state_dict = {
            "layer.weight": torch.ones((16, 16), dtype=torch.float16).to(
                torch.float8_e4m3fn
            ),
            "layer.weight_scale": torch.tensor(2.0, dtype=torch.float32),
            "layer.comfy_quant": core.encode_quant_config(
                {"format": "float8_e4m3fn"}
            ),
        }
        quantization = {
            "format_version": "1.0",
            "layers": {"layer": {"format": "float8_e4m3fn"}},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "model-fp8.safetensors")
            safetensors.torch.save_file(
                state_dict,
                source,
                metadata={
                    "_quantization_metadata": json.dumps(quantization),
                    "fixture": "streaming-fp8",
                },
            )

            _, output = core.convert_model(
                source, "Other/Unknown", "fp16", "cpu"
            )

            converted = safetensors.torch.load_file(output)
            self.assertEqual(set(converted), {"layer.weight"})
            self.assertEqual(converted["layer.weight"].dtype, torch.float16)
            self.assertTrue(
                torch.equal(
                    converted["layer.weight"],
                    torch.full((16, 16), 2.0, dtype=torch.float16),
                )
            )
            with safetensors.safe_open(output, framework="pt") as handle:
                metadata = handle.metadata()
            self.assertEqual(metadata["fixture"], "streaming-fp8")
            self.assertNotIn("_quantization_metadata", metadata)

    def test_streaming_int8_input_applies_weight_scale_before_fp16_conversion(self):
        state_dict = {
            "layer.weight": torch.ones((16, 16), dtype=torch.int8),
            "layer.weight_scale": torch.tensor(3.0, dtype=torch.float32),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "model-int8.safetensors")
            safetensors.torch.save_file(state_dict, source)

            _, output = core.convert_model(
                source, "Other/Unknown", "fp16", "cpu"
            )

            converted = safetensors.torch.load_file(output)
            self.assertEqual(set(converted), {"layer.weight"})
            self.assertEqual(converted["layer.weight"].dtype, torch.float16)
            self.assertTrue(
                torch.equal(
                    converted["layer.weight"],
                    torch.full((16, 16), 3.0, dtype=torch.float16),
                )
            )

    def test_streaming_int8_input_without_scale_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "model-int8.safetensors")
            safetensors.torch.save_file(
                {"layer.weight": torch.ones((16, 16), dtype=torch.int8)},
                source,
            )

            with self.assertRaisesRegex(ValueError, "has no 'layer.weight_scale'"):
                core.convert_model(source, "Other/Unknown", "fp16", "cpu")

            self.assertFalse(os.path.exists(os.path.join(temp_dir, "model-fp16.safetensors")))

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "PyTorch has no FP8 dtype")
    def test_streaming_scaled_fp8_removes_auxiliary_tensors(self):
        state_dict = {
            "scaled_fp8": torch.tensor(1, dtype=torch.uint8),
            "layer.weight": torch.ones((16, 16), dtype=torch.float16).to(
                torch.float8_e4m3fn
            ),
            "layer.scale_weight": torch.tensor(4.0, dtype=torch.float32),
            "layer.scale_input": torch.tensor(5.0, dtype=torch.float32),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "model-fp8-scaled.safetensors")
            safetensors.torch.save_file(state_dict, source)

            status, output = core.convert_model(
                source, "Other/Unknown", "fp16", "cpu"
            )

            converted = safetensors.torch.load_file(output)
            self.assertEqual(set(converted), {"layer.weight"})
            self.assertTrue(
                torch.equal(
                    converted["layer.weight"],
                    torch.full((16, 16), 4.0, dtype=torch.float16),
                )
            )
            self.assertIn("[ComfyUI scaled fp8]", status)

    def test_streaming_rejects_lossy_embedded_quantization_before_writing(self):
        state_dict = {
            "layer.weight": torch.ones((16, 16), dtype=torch.float16),
            "layer.comfy_quant": core.encode_quant_config({"format": "nvfp4"}),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "model-nvfp4.safetensors")
            safetensors.torch.save_file(state_dict, source)

            with self.assertRaisesRegex(ValueError, "cannot be dequantized losslessly"):
                core.convert_model(source, "Other/Unknown", "fp16", "cpu")

            self.assertFalse(os.path.exists(os.path.join(temp_dir, "model-fp16.safetensors")))

    def test_atomic_save_replaces_existing_output_after_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "model-bf16.safetensors")
            output = os.path.join(temp_dir, "model-fp16.safetensors")
            safetensors.torch.save_file(
                {"layer.weight": torch.full((16, 16), 2.0, dtype=torch.bfloat16)},
                source,
            )
            safetensors.torch.save_file(
                {"stale": torch.tensor(1, dtype=torch.int64)},
                output,
                metadata={"state": "old"},
            )

            _, converted_path = core.convert_model(
                source, "Other/Unknown", "fp16", "cpu"
            )

            self.assertEqual(converted_path, output)
            converted = safetensors.torch.load_file(output)
            self.assertEqual(set(converted), {"layer.weight"})
            self.assertFalse(any(name.endswith(".partial") for name in os.listdir(temp_dir)))

    def test_atomic_save_validation_failure_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "model-bf16.safetensors")
            output = os.path.join(temp_dir, "model-fp16.safetensors")
            safetensors.torch.save_file(
                {"layer.weight": torch.ones((16, 16), dtype=torch.bfloat16)},
                source,
            )
            safetensors.torch.save_file(
                {"sentinel": torch.tensor(7, dtype=torch.int64)},
                output,
                metadata={"state": "old"},
            )

            with mock.patch.object(
                core,
                "_validate_saved_safetensors",
                side_effect=RuntimeError("validation failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "validation failed"):
                    core.convert_model(source, "Other/Unknown", "fp16", "cpu")

            preserved = safetensors.torch.load_file(output)
            self.assertEqual(set(preserved), {"sentinel"})
            self.assertEqual(preserved["sentinel"].item(), 7)
            self.assertFalse(any(name.endswith(".partial") for name in os.listdir(temp_dir)))

    def test_atomic_save_write_failure_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "model-bf16.safetensors")
            output = os.path.join(temp_dir, "model-fp16.safetensors")
            safetensors.torch.save_file(
                {"layer.weight": torch.ones((16, 16), dtype=torch.bfloat16)},
                source,
            )
            safetensors.torch.save_file(
                {"sentinel": torch.tensor(9, dtype=torch.int64)},
                output,
                metadata={"state": "old"},
            )

            with mock.patch.object(
                core.safetensors.torch,
                "save_file",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    core.convert_model(source, "Other/Unknown", "fp16", "cpu")

            preserved = safetensors.torch.load_file(output)
            self.assertEqual(set(preserved), {"sentinel"})
            self.assertEqual(preserved["sentinel"].item(), 9)
            self.assertFalse(any(name.endswith(".partial") for name in os.listdir(temp_dir)))


class UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = types.ModuleType("modules")
        modules.call_queue = types.SimpleNamespace(wrap_queued_call=lambda fn: fn)
        modules.paths = types.SimpleNamespace(models_path="")
        modules.script_callbacks = types.SimpleNamespace(on_ui_tabs=lambda *_args, **_kwargs: None)
        modules.shared = types.SimpleNamespace(
            cmd_opts=types.SimpleNamespace(ckpt_dirs=[], text_encoder_dirs=[])
        )
        modules.ui = types.SimpleNamespace(refresh_symbol="Refresh")

        ui_components = types.ModuleType("modules.ui_components")

        def tool_button(**kwargs):
            kwargs.pop("tooltip", None)
            return gr.Button(**kwargs)

        ui_components.ToolButton = tool_button

        cls.modules_patch = mock.patch.dict(
            sys.modules,
            {
                "modules": modules,
                "modules.ui_components": ui_components,
            },
        )
        cls.modules_patch.start()

        spec = importlib.util.spec_from_file_location("forge_neo_converter_ui_tests", UI_SCRIPT)
        cls.converter_ui = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.converter_ui)

    @classmethod
    def tearDownClass(cls):
        cls.modules_patch.stop()

    def test_text_encoder_scanner_uses_forge_and_configured_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            models_root = os.path.join(temp_dir, "models")
            native_dir = os.path.join(models_root, "text_encoder")
            extra_dir = os.path.join(temp_dir, "extra_text_encoders")
            os.makedirs(native_dir)
            os.makedirs(extra_dir)

            native_file = os.path.join(native_dir, "clip_l.safetensors")
            extra_file = os.path.join(extra_dir, "t5", "t5xxl.safetensors")
            os.makedirs(os.path.dirname(extra_file))
            Path(native_file).touch()
            Path(extra_file).touch()

            self.converter_ui.paths.models_path = models_root
            self.converter_ui.shared.cmd_opts.text_encoder_dirs = [extra_dir]

            choices = self.converter_ui.list_text_encoder_choices()

            self.assertEqual(choices, ["clip_l.safetensors", "t5/t5xxl.safetensors"])
            self.assertEqual(
                self.converter_ui.resolve_text_encoder_path("clip_l.safetensors"),
                native_file,
            )

    def test_converter_ui_builds_both_modes(self):
        with tempfile.TemporaryDirectory() as models_root:
            self.converter_ui.paths.models_path = models_root
            self.converter_ui.shared.cmd_opts.ckpt_dirs = []
            self.converter_ui.shared.cmd_opts.text_encoder_dirs = []

            tabs = self.converter_ui.create_converter_tab()

            self.assertEqual([(title, elem_id) for _, title, elem_id in tabs], [
                ("Converter", "forge_neo_converter_tab")
            ])
            config = tabs[0][0].get_config_file()
            tab_labels = {
                component["props"].get("label")
                for component in config["components"]
                if component["type"] == "tabitem"
            }
            self.assertEqual(tab_labels, {"Model mode", "Text encoder mode"})

    def test_ui_recovers_from_stale_cached_core_module(self):
        stale_core = types.ModuleType("forge_neo_converter_core")
        stale_core.__file__ = str(ROOT / "forge_neo_converter_core.py")
        stale_core.TARGET_FORMATS = ["fp16"]

        with mock.patch.dict(sys.modules, {"forge_neo_converter_core": stale_core}):
            spec = importlib.util.spec_from_file_location(
                "forge_neo_converter_stale_core_test",
                UI_SCRIPT,
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            self.assertTrue(hasattr(module.converter_core, "convert_text_encoder"))
            self.assertEqual(module.TEXT_ENCODER_PROFILE, "Text-Encoder")
            self.assertIn("w4a8_convrot", module.TARGET_FORMATS)


if __name__ == "__main__":
    unittest.main()
