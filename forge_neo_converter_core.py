import json
import os
import gc
import re
import time
from collections import Counter, OrderedDict

import safetensors
import safetensors.torch
import torch

try:
    import comfy_kitchen as ck
    from comfy_kitchen.registry import registry as ck_registry
    from comfy_kitchen.tensor import (
        TensorCoreConvRotW4A4Layout,
        TensorCoreMXFP8Layout,
        TensorCoreNVFP4Layout,
        TensorWiseINT8Layout,
    )
except ImportError:
    ck = None
    ck_registry = None
    TensorCoreMXFP8Layout = None
    TensorCoreNVFP4Layout = None
    TensorWiseINT8Layout = None
    TensorCoreConvRotW4A4Layout = None

try:
    from comfy_kitchen.tensor import AsymW4A8Int8Layout
except ImportError:
    AsymW4A8Int8Layout = None


EXTENSION_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_JSON = os.path.join(EXTENSION_DIR, "models.json")

EXTENDED_METADATA_KEYS = ["config", "license", "encrypted_wandb_properties"]
TARGET_FORMATS = ["nvfp4", "fp8", "mxfp8", "int8", "int8_convrot", "int4_convrot", "w4a8_convrot", "fp16", "fp32"]
TEXT_ENCODER_PROFILE = "Text-Encoder"
CONVROT_GROUPSIZE = 256
INT4_QUANT_GROUPSIZE = 64
W4A8_QUANT_GROUPSIZE = 16
FORGE_SENSITIVE_SUBSTRINGS = (
    "embed",
    "bias",
    "norm",
    "scale",
    "llm",
    "first_stage_model",
    "cond_stage_model",
    "vae",
    "text",
    "time",
)

PRECISION_RE = re.compile(
    r"[-_.](fp32|fp16|bf16|mxfp8|fp8(?:_e[45]m[23](?:fn)?)?(?:_scaled)?(?:_fast)?|int[48](?:_convrot)?|w4a[48]_convrot|nvfp4)(?=[-_.]|$)",
    re.IGNORECASE,
)

FP8_DTYPES = tuple(dtype for dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)) if dtype is not None)
FLOAT8_E8M0 = getattr(torch, "float8_e8m0fnu", None)

DTYPE_NAMES = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.int8: "int8",
}

if hasattr(torch, "float8_e4m3fn"):
    DTYPE_NAMES[torch.float8_e4m3fn] = "fp8_e4m3fn"
if hasattr(torch, "float8_e5m2"):
    DTYPE_NAMES[torch.float8_e5m2] = "fp8_e5m2"


def _noop_logger(_message):
    pass


def load_model_configs():
    with open(MODELS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def model_types():
    return list(load_model_configs()["models"].keys())


def get_profile(configs, model_type):
    default = configs["default"]
    profile = configs["models"].get(model_type, default)
    return (
        profile.get("blacklist", default["blacklist"]),
        profile.get("fp8_layers", default["fp8_layers"]),
        profile.get("preserve_extended_metadata", default["preserve_extended_metadata"]),
    )


def build_output_path(out_dir, base_name, target_format):
    stem = PRECISION_RE.sub("", base_name).rstrip("-_.")
    return os.path.join(out_dir, f"{stem}-{target_format}.safetensors")


def format_size(num_bytes):
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def encode_quant_config(info):
    return torch.tensor(list(json.dumps(info).encode("utf-8")), dtype=torch.uint8)


def keep_tensor_dtype(tensor):
    if tensor.dtype in (torch.float32, torch.bfloat16):
        return tensor.to(dtype=torch.float16)
    return tensor


def preserve_tensor(tensor, source_kind):
    if source_kind == "text_encoder" and tensor.dtype.is_floating_point:
        return tensor.to(dtype=torch.bfloat16), "kept bf16"
    return keep_tensor_dtype(tensor), "kept"


def can_quantize_weight(key, tensor, protected_substrings=FORGE_SENSITIVE_SUBSTRINGS, alignment=16):
    if not key.endswith(".weight"):
        return False
    if any(name in key for name in protected_substrings):
        return False
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if tensor.ndim != 2:
        return False
    return tensor.size(0) % alignment == 0 and tensor.size(1) % alignment == 0


def detect_input_format(sd, metadata):
    counts = Counter(DTYPE_NAMES.get(v.dtype, str(v.dtype)) for v in sd.values())
    parts = [f"{name} ({n} tensors)" for name, n in counts.most_common()]
    fmt = ", ".join(parts)
    if "scaled_fp8" in sd:
        fmt += " [ComfyUI scaled fp8]"
    elif metadata and "_quantization_metadata" in metadata:
        fmt += " [quantization metadata]"
    return fmt


def load_input(path, log=_noop_logger):
    log(f"Loading: {path}")
    sd = safetensors.torch.load_file(path)
    with safetensors.safe_open(path, framework="pt") as f:
        orig_meta = f.metadata()
    return sd, orig_meta


def inspect_lazy_input(path, log=_noop_logger):
    """Inspect a safetensors file without retaining the whole state dict in RAM."""
    log(f"Inspecting lazily: {path}")
    counts = Counter()
    with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        metadata = f.metadata()
        for k in keys:
            t = f.get_tensor(k)
            counts[DTYPE_NAMES.get(t.dtype, str(t.dtype))] += 1
            del t
    gc.collect()
    parts = [f"{name} ({n} tensors)" for name, n in counts.most_common()]
    fmt = ", ".join(parts)
    if metadata and "_quantization_metadata" in metadata:
        fmt += " [quantization metadata]"
    return keys, metadata, fmt


def can_stream_input(metadata, keys):
    """Return True for normal high-precision inputs that need no cross-tensor dequantization."""
    if metadata and "_quantization_metadata" in metadata:
        return False
    return "scaled_fp8" not in keys


def dequantize_input(sd, metadata, log=_noop_logger):
    quant_layers = {}
    if metadata and "_quantization_metadata" in metadata:
        quant_layers = json.loads(metadata["_quantization_metadata"]).get("layers", {})

    for k in [k for k in sd if k.endswith(".comfy_quant")]:
        conf = sd.pop(k)
        layer = k[: -len(".comfy_quant")]
        if layer not in quant_layers:
            try:
                quant_layers[layer] = json.loads(bytes(conf.cpu().to(torch.uint8).tolist()))
            except Exception:
                log(f"Warning: could not parse embedded quant config for '{layer}', ignoring.")

    for layer, info in quant_layers.items():
        fmt = info.get("format")
        if fmt in ("nvfp4", "mxfp8", "convrot_w4a4", "asym_w4a8_int8"):
            raise ValueError(f"Input model contains {fmt} layers ('{layer}'), which cannot be dequantized losslessly. Use a higher precision source model.")
        if info.get("convrot") and fmt != "convrot_w4a4":
            raise ValueError(f"Input model contains ConvRot-rotated INT8 layers ('{layer}'). Use a higher precision source model.")

    if "scaled_fp8" in sd:
        sd.pop("scaled_fp8")
        for k in [k for k in sd if k.endswith(".scale_weight")]:
            scale = sd.pop(k)
            wk = k[: -len(".scale_weight")] + ".weight"
            if wk in sd:
                sd[wk] = (sd[wk].to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)
        for k in [k for k in sd if k.endswith(".scale_input")]:
            sd.pop(k)

    for k in list(sd.keys()):
        if k not in sd or not k.endswith(".weight"):
            continue
        v = sd[k]
        if v.dtype in FP8_DTYPES or v.dtype == torch.int8:
            scale = sd.pop(k + "_scale", None)
            if scale is not None:
                sd[k] = (v.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)
            elif v.dtype == torch.int8:
                raise ValueError(f"int8 weight '{k}' has no '{k}_scale' tensor, cannot dequantize.")

    return sd


def pick_mxfp8_backend(device, log=_noop_logger):
    probe = torch.randn(32, 32, device=device, dtype=torch.float32)
    try:
        TensorCoreMXFP8Layout.quantize(probe)
        return None
    except Exception as e:
        log(f"Warning: MXFP8 default backend failed ({e}). Trying fallback backends...")
    for backend in ("triton", "eager"):
        try:
            with ck_registry.use_backend(backend):
                TensorCoreMXFP8Layout.quantize(probe)
            log(f"MXFP8: using '{backend}' backend")
            return backend
        except Exception:
            continue
    raise RuntimeError("MXFP8 quantization is not supported by any comfy_kitchen backend in this environment. Try updating comfy-kitchen and PyTorch.")


def _require_quantization_deps(target_format):
    if target_format in ("fp16", "fp32"):
        return
    if ck is None:
        raise RuntimeError("comfy-kitchen is not installed. Install extension requirements before using quantized target formats.")
    if target_format == "int4_convrot" and TensorCoreConvRotW4A4Layout is None:
        raise RuntimeError("This comfy-kitchen version does not support int4_convrot. Update Forge Neo/comfy-kitchen first.")
    if target_format == "w4a8_convrot" and AsymW4A8Int8Layout is None:
        raise RuntimeError("This comfy-kitchen version does not support w4a8_convrot. Update Forge Neo/comfy-kitchen first.")


def convert_model(model_path, model_type, target_format, device, log=_noop_logger, source_kind="model"):
    if not model_path:
        raise ValueError("No model selected.")
    if not os.path.isfile(model_path):
        raise ValueError(f"Model file not found: {model_path}")
    if source_kind not in ("model", "text_encoder"):
        raise ValueError(f"Unsupported source kind: {source_kind}")
    if target_format not in TARGET_FORMATS:
        raise ValueError(f"Unsupported target format: {target_format}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but torch.cuda.is_available() is false.")

    _require_quantization_deps(target_format)

    configs = load_model_configs()
    active_model_type = TEXT_ENCODER_PROFILE if source_kind == "text_encoder" else model_type
    blacklist, fp8_layers, preserve_extended = get_profile(configs, active_model_type)
    protected_substrings = () if source_kind == "text_encoder" else FORGE_SENSITIVE_SUBSTRINGS
    source_label = "Text encoder" if source_kind == "text_encoder" else "Model"
    start_time = time.time()
    out_dir = os.path.dirname(model_path)
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    output_path = build_output_path(out_dir, base_name, target_format)

    log(f"{source_label} conversion profile: {active_model_type} | target: {target_format}")
    log(f"Converting on: {device}")

    input_bytes = os.path.getsize(model_path)
    lazy_keys, orig_meta, lazy_format = inspect_lazy_input(model_path, log=log)
    streaming_input = can_stream_input(orig_meta, lazy_keys)

    if streaming_input:
        log("Memory mode: streaming input tensors (avoids loading the full source model into RAM)")
        sd = None
        input_format = lazy_format
        total = len(lazy_keys)
    else:
        log("Memory mode: legacy full input load (input requires dequantization or special handling)")
        sd, orig_meta = load_input(model_path, log=log)
        input_format = detect_input_format(sd, orig_meta)
        sd = dequantize_input(sd, orig_meta, log=log)
        total = len(sd)
        lazy_keys = list(sd.keys())

    temp_diffusers_meta = OrderedDict()
    if orig_meta:
        for key, value in orig_meta.items():
            if key != "_quantization_metadata":
                temp_diffusers_meta[key] = value

    log(f"Original format: {input_format}")

    quant_map = {"format_version": "1.0", "layers": {}}
    new_sd = {}
    counts = Counter()
    mxfp8_backend = pick_mxfp8_backend(device, log=log) if target_format == "mxfp8" else None
    quant_alignment = CONVROT_GROUPSIZE if target_format == "w4a8_convrot" else 16

    def iter_input_tensors():
        if streaming_input:
            with safetensors.safe_open(model_path, framework="pt", device="cpu") as handle:
                for key in lazy_keys:
                    yield key, handle.get_tensor(key)
        else:
            yield from sd.items()

    if target_format in ("fp16", "fp32"):
        target_dtype = torch.float16 if target_format == "fp16" else torch.float32
        for i, (k, v) in enumerate(iter_input_tensors(), start=1):
            if i == 1 or i == total or i % 100 == 0:
                log(f"Progress: {i}/{total}")
            if v.dtype.is_floating_point:
                new_sd[k] = v.to(target_dtype)
                counts[target_format] += 1
            else:
                new_sd[k] = v
                counts["kept"] += 1
            if i % 16 == 0:
                gc.collect()
    else:
        for i, (k, v) in enumerate(iter_input_tensors(), start=1):
            if i == 1 or i == total or i % 100 == 0:
                log(f"Progress: {i}/{total}")

            if any(name in k for name in blacklist):
                new_sd[k], count_name = preserve_tensor(v, source_kind)
                counts[count_name] += 1
                del v
                continue

            if can_quantize_weight(
                k,
                v,
                protected_substrings=protected_substrings,
                alignment=quant_alignment,
            ):
                base_k_file = k.replace(".weight", "")
                base_k_meta = base_k_file

                # Current Forge kitchen kernels accept FP16/BF16 input only.  Keeping
                # this tensor in BF16 also avoids creating a large FP32 copy per layer.
                v_tensor = v.to(device=device, dtype=torch.bfloat16)

                if target_format == "fp8" or (fp8_layers and any(name in k for name in fp8_layers)):
                    log(f"FP8: {k}")
                    weight_scale = (v_tensor.abs().max() / 448.0).clamp(min=1e-12).float()
                    weight_quantized = ck.quantize_per_tensor_fp8(v_tensor, weight_scale)
                    new_sd[k] = weight_quantized.cpu()
                    new_sd[f"{base_k_file}.weight_scale"] = weight_scale.to(torch.bfloat16).cpu()
                    layer_conf = {"format": "float8_e4m3fn"}
                    new_sd[f"{base_k_file}.comfy_quant"] = encode_quant_config(layer_conf)
                    quant_map["layers"][base_k_meta] = layer_conf
                    counts["fp8"] += 1
                    del v_tensor, v
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    continue

                int8_convrot = target_format == "int8_convrot"
                int4_convrot = target_format == "int4_convrot"
                w4a8_convrot = target_format == "w4a8_convrot"
                if target_format in ("int8", "int8_convrot"):
                    layout = TensorWiseINT8Layout
                    fmt_name = "int8_tensorwise"
                elif int4_convrot:
                    layout = TensorCoreConvRotW4A4Layout
                    fmt_name = "convrot_w4a4"
                elif w4a8_convrot:
                    layout = AsymW4A8Int8Layout
                    fmt_name = "asym_w4a8_int8"
                elif target_format == "mxfp8":
                    layout = TensorCoreMXFP8Layout
                    fmt_name = "mxfp8"
                else:
                    layout = TensorCoreNVFP4Layout
                    fmt_name = "nvfp4"
                log(f"{target_format.upper()}: {k}")

                qdata = params = tensors = v_tensor_ready = None
                try:
                    # Do not cast to float32 here: recent Forge kernels reject it
                    # ("Unsupported dtype code: 0") and the temporary FP32 copy can
                    # exhaust VRAM on large DiTs.
                    v_tensor_ready = v_tensor.contiguous()
                    if int8_convrot:
                        qdata, params = layout.quantize(v_tensor_ready, per_channel=True, convrot=True, convrot_groupsize=CONVROT_GROUPSIZE)
                    elif int4_convrot:
                        qdata, params = layout.quantize(
                            v_tensor_ready,
                            convrot_groupsize=CONVROT_GROUPSIZE,
                            quant_group_size=INT4_QUANT_GROUPSIZE,
                        )
                    elif w4a8_convrot:
                        qdata, params = layout.quantize(
                            v_tensor_ready,
                            group_size=W4A8_QUANT_GROUPSIZE,
                            convrot_groupsize=CONVROT_GROUPSIZE,
                            scale_dtype=torch.float8_e4m3fn,
                        )
                    elif mxfp8_backend is not None:
                        with ck_registry.use_backend(mxfp8_backend):
                            qdata, params = layout.quantize(v_tensor_ready)
                    else:
                        qdata, params = layout.quantize(v_tensor_ready)

                    tensors = layout.state_dict_tensors(qdata, params)
                    for suffix, tensor in tensors.items():
                        out_key = f"{base_k_file}.weight{suffix}"
                        if FLOAT8_E8M0 is not None and tensor.dtype == FLOAT8_E8M0:
                            new_sd[out_key] = tensor.view(torch.uint8).cpu()
                        elif tensor.dtype in FP8_DTYPES:
                            new_sd[out_key] = tensor.view(torch.uint8).cpu().view(tensor.dtype)
                        else:
                            new_sd[out_key] = tensor.cpu()

                    layer_conf = {"format": fmt_name}
                    if int8_convrot:
                        layer_conf["convrot"] = True
                        layer_conf["convrot_groupsize"] = CONVROT_GROUPSIZE
                    elif int4_convrot:
                        layer_conf["convrot_groupsize"] = CONVROT_GROUPSIZE
                        layer_conf["quant_group_size"] = INT4_QUANT_GROUPSIZE
                    elif w4a8_convrot:
                        layer_conf["group_size"] = W4A8_QUANT_GROUPSIZE
                        layer_conf["convrot_groupsize"] = CONVROT_GROUPSIZE
                    new_sd[f"{base_k_file}.comfy_quant"] = encode_quant_config(layer_conf)
                    quant_map["layers"][base_k_meta] = layer_conf
                    counts[target_format] += 1
                except Exception as e:
                    log(f"Warning: quantization failed for {k}: {e}")
                    new_sd[k], count_name = preserve_tensor(v, source_kind)
                    counts[count_name] += 1

                # Explicitly drop all CUDA temporaries before the next layer.  The
                # quantization layouts can allocate several working buffers, so
                # retaining one iteration is enough to OOM a 16 GB GPU.
                del qdata, params, tensors, v_tensor_ready, v_tensor
                del v
                if device == "cuda":
                    torch.cuda.empty_cache()
            else:
                new_sd[k], count_name = preserve_tensor(v, source_kind)
                counts[count_name] += 1
                del v
            if i % 16 == 0:
                gc.collect()

    final_metadata = OrderedDict(temp_diffusers_meta)
    if quant_map["layers"]:
        final_metadata["_quantization_metadata"] = json.dumps(quant_map)
        first_quant_layer = next(iter(quant_map["layers"]))
        log(f"Quantization metadata: {len(quant_map['layers'])} layers, first key: {first_quant_layer}")
    final_metadata["converted_by"] = "Star Ultimate Model Converter (streaming input patch)"

    log(f"Saving | Type: {active_model_type} | Path: {output_path}")
    safetensors.torch.save_file(new_sd, output_path, metadata=final_metadata)

    output_bytes = os.path.getsize(output_path)
    duration = time.time() - start_time
    reduction = (1 - output_bytes / input_bytes) * 100 if input_bytes else 0
    layers_desc = ", ".join(f"{n} {name}" for name, n in counts.most_common())
    status = "\n".join(
        [
            f"Success ({active_model_type} -> {target_format})",
            f"Input: {os.path.basename(model_path)}",
            f"Original format: {input_format}",
            f"Original size: {format_size(input_bytes)}",
            f"New size: {format_size(output_bytes)} ({reduction:.1f}% smaller)",
            f"Layers: {layers_desc}",
            f"Device: {device} | Time: {duration:.1f}s",
            f"Saved to: {output_path}",
        ]
    )
    log(status)
    return status, output_path


def convert_text_encoder(model_path, target_format, device, log=_noop_logger):
    if not model_path:
        raise ValueError("No text encoder selected.")
    return convert_model(
        model_path,
        TEXT_ENCODER_PROFILE,
        target_format,
        device,
        log=log,
        source_kind="text_encoder",
    )
