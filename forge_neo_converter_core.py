import json
import os
import re
import tempfile
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass

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
CONVROT_GROUPSIZES = (256, 64, 16)
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


@dataclass
class StreamingInputPlan:
    keys: list
    metadata: dict
    input_format: str
    dtypes: dict
    scale_by_weight: dict
    skipped_keys: set


def _noop_logger(_message):
    pass


def load_model_configs():
    with open(MODELS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def model_types():
    return list(load_model_configs()["models"].keys())


def get_profile(configs, model_type, target_format=None):
    default = configs["default"]
    profile = configs["models"].get(model_type, default)
    blacklist = list(profile.get("blacklist", default["blacklist"]))
    override = {}
    if target_format:
        override = profile.get("target_overrides", {}).get(target_format, {})
        remove = set(override.get("blacklist_remove", []))
        if remove:
            blacklist = [name for name in blacklist if name not in remove]
        for name in override.get("blacklist_add", []):
            if name not in blacklist:
                blacklist.append(name)
    return (
        blacklist,
        profile.get("fp8_layers", default["fp8_layers"]),
        profile.get("preserve_extended_metadata", default["preserve_extended_metadata"]),
        override,
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


def is_quantizable_weight(key, tensor, protected_substrings=FORGE_SENSITIVE_SUBSTRINGS):
    if not key.endswith(".weight"):
        return False
    if any(name in key for name in protected_substrings):
        return False
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    return tensor.ndim == 2


def can_quantize_weight(key, tensor, protected_substrings=FORGE_SENSITIVE_SUBSTRINGS, alignment=16):
    if not is_quantizable_weight(key, tensor, protected_substrings=protected_substrings):
        return False
    return tensor.size(0) % alignment == 0 and tensor.size(1) % alignment == 0


def best_convrot_groupsize(in_features):
    """Pick the largest supported Hadamard group that divides K (in_features)."""
    return next((group for group in CONVROT_GROUPSIZES if in_features % group == 0), None)


def tensor_nbytes(tensor):
    return tensor.numel() * tensor.element_size()


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


def _parse_embedded_quant_config(tensor):
    return json.loads(bytes(tensor.cpu().to(torch.uint8).tolist()))


def _validate_quant_layers(quant_layers):
    for layer, info in quant_layers.items():
        fmt = info.get("format")
        if fmt in ("nvfp4", "mxfp8", "convrot_w4a4", "asym_w4a8_int8"):
            raise ValueError(
                f"Input model contains {fmt} layers ('{layer}'), which cannot be "
                "dequantized losslessly. Use a higher precision source model."
            )
        if info.get("convrot") and fmt != "convrot_w4a4":
            raise ValueError(
                f"Input model contains ConvRot-rotated INT8 layers ('{layer}'). "
                "Use a higher precision source model."
            )


def inspect_streaming_input(path, log=_noop_logger):
    """Build a lightweight plan for reading and dequantizing one tensor at a time."""
    log(f"Inspecting input: {path}")
    counts = Counter()
    dtypes = {}
    embedded_quant_layers = {}

    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        metadata = handle.metadata()
        for key in keys:
            tensor = handle.get_tensor(key)
            dtypes[key] = tensor.dtype
            counts[DTYPE_NAMES.get(tensor.dtype, str(tensor.dtype))] += 1
            if key.endswith(".comfy_quant"):
                layer = key[: -len(".comfy_quant")]
                try:
                    embedded_quant_layers[layer] = _parse_embedded_quant_config(tensor)
                except Exception:
                    log(f"Warning: could not parse embedded quant config for '{layer}', ignoring.")

    quant_layers = {}
    if metadata and "_quantization_metadata" in metadata:
        quant_layers = json.loads(metadata["_quantization_metadata"]).get("layers", {})
    for layer, info in embedded_quant_layers.items():
        quant_layers.setdefault(layer, info)
    _validate_quant_layers(quant_layers)

    skipped_keys = {key for key in keys if key.endswith(".comfy_quant")}
    scale_by_weight = {}

    if "scaled_fp8" in dtypes:
        skipped_keys.add("scaled_fp8")
        skipped_keys.update(key for key in keys if key.endswith(".scale_input"))
        for scale_key in (key for key in keys if key.endswith(".scale_weight")):
            weight_key = scale_key[: -len(".scale_weight")] + ".weight"
            if weight_key in dtypes:
                scale_by_weight[weight_key] = scale_key
                skipped_keys.add(scale_key)

    for key, dtype in dtypes.items():
        if not key.endswith(".weight") or key in scale_by_weight:
            continue
        if dtype in FP8_DTYPES or dtype == torch.int8:
            scale_key = key + "_scale"
            if scale_key in dtypes:
                scale_by_weight[key] = scale_key
                skipped_keys.add(scale_key)
            elif dtype == torch.int8:
                raise ValueError(
                    f"int8 weight '{key}' has no '{scale_key}' tensor, cannot dequantize."
                )

    parts = [f"{name} ({n} tensors)" for name, n in counts.most_common()]
    input_format = ", ".join(parts)
    if "scaled_fp8" in dtypes:
        input_format += " [ComfyUI scaled fp8]"
    elif metadata and "_quantization_metadata" in metadata:
        input_format += " [quantization metadata]"

    return StreamingInputPlan(
        keys=[key for key in keys if key not in skipped_keys],
        metadata=metadata,
        input_format=input_format,
        dtypes=dtypes,
        scale_by_weight=scale_by_weight,
        skipped_keys=skipped_keys,
    )


def iter_streaming_input(path, plan):
    """Yield input tensors while applying lossless per-tensor dequantization."""
    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        for key in plan.keys:
            tensor = handle.get_tensor(key)
            scale_key = plan.scale_by_weight.get(key)
            if scale_key is not None:
                scale = handle.get_tensor(scale_key)
                tensor = (tensor.to(torch.float32) * scale.to(torch.float32)).to(
                    torch.bfloat16
                )
            yield key, tensor


def _validate_saved_safetensors(path, tensors, metadata):
    if os.path.getsize(path) <= 0:
        raise RuntimeError("Saved safetensors file is empty.")

    expected_keys = set(tensors)
    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        saved_keys = set(handle.keys())
        if saved_keys != expected_keys:
            missing = sorted(expected_keys - saved_keys)
            extra = sorted(saved_keys - expected_keys)
            raise RuntimeError(
                f"Saved safetensors keys do not match output. Missing: {missing}; "
                f"extra: {extra}."
            )

        for key, expected in tensors.items():
            saved = handle.get_tensor(key)
            if saved.dtype != expected.dtype or tuple(saved.shape) != tuple(expected.shape):
                raise RuntimeError(
                    f"Saved tensor '{key}' does not match output: expected "
                    f"{expected.dtype} {tuple(expected.shape)}, got "
                    f"{saved.dtype} {tuple(saved.shape)}."
                )

        saved_metadata = handle.metadata() or {}

    expected_metadata = dict(metadata or {})
    if saved_metadata != expected_metadata:
        raise RuntimeError("Saved safetensors metadata does not match output metadata.")


def save_safetensors_atomic(tensors, output_path, metadata, log=_noop_logger):
    """Validate a same-directory temporary file before atomically replacing output."""
    output_dir = os.path.dirname(os.path.abspath(output_path))
    output_name = os.path.basename(output_path)
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=output_dir,
            prefix=f".{output_name}.",
            suffix=".partial",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name

        log(f"Writing temporary output: {temp_path}")
        safetensors.torch.save_file(tensors, temp_path, metadata=metadata)
        _validate_saved_safetensors(temp_path, tensors, metadata)
        os.replace(temp_path, output_path)
        temp_path = None
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as error:
                log(f"Warning: could not remove temporary output '{temp_path}': {error}")


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

    _validate_quant_layers(quant_layers)

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
    blacklist, fp8_layers, preserve_extended, target_override = get_profile(
        configs, active_model_type, target_format=target_format
    )
    protected_substrings = () if source_kind == "text_encoder" else FORGE_SENSITIVE_SUBSTRINGS
    source_label = "Text encoder" if source_kind == "text_encoder" else "Model"
    start_time = time.time()
    out_dir = os.path.dirname(model_path)
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    output_path = build_output_path(out_dir, base_name, target_format)

    log(f"{source_label} conversion profile: {active_model_type} | target: {target_format}")
    if target_override:
        removed = target_override.get("blacklist_remove", [])
        added = target_override.get("blacklist_add", [])
        details = []
        if removed:
            details.append("allow=" + ",".join(removed))
        if added:
            details.append("protect=" + ",".join(added))
        if details:
            log("Target-specific profile override: " + " | ".join(details))
    log(f"Converting on: {device}")

    input_bytes = os.path.getsize(model_path)
    input_plan = inspect_streaming_input(model_path, log=log)
    orig_meta = input_plan.metadata

    temp_diffusers_meta = OrderedDict()
    if orig_meta:
        for key, value in orig_meta.items():
            if key != "_quantization_metadata":
                temp_diffusers_meta[key] = value

    input_format = input_plan.input_format
    log(f"Original format: {input_format}")
    log("Memory mode: streaming source tensors with per-layer dequantization")

    quant_map = {"format_version": "1.0", "layers": {}}
    new_sd = {}
    counts = Counter()
    reason_counts = Counter()
    coverage_bytes = Counter()
    profile_keep_bytes = Counter()
    global_keep_bytes = Counter()
    matrix_source_bytes = 0
    total = len(input_plan.keys)
    mxfp8_backend = pick_mxfp8_backend(device, log=log) if target_format == "mxfp8" else None
    convrot_target = target_format in ("int8_convrot", "int4_convrot", "w4a8_convrot")
    quant_alignment = 16

    if target_format in ("fp16", "fp32"):
        target_dtype = torch.float16 if target_format == "fp16" else torch.float32
        for i, (k, v) in enumerate(iter_streaming_input(model_path, input_plan), start=1):
            if i == 1 or i == total or i % 100 == 0:
                log(f"Progress: {i}/{total}")
            if v.dtype.is_floating_point:
                new_sd[k] = v.to(target_dtype)
                counts[target_format] += 1
            else:
                new_sd[k] = v
                counts["kept"] += 1
    else:
        for i, (k, v) in enumerate(iter_streaming_input(model_path, input_plan), start=1):
            if i == 1 or i == total or i % 100 == 0:
                log(f"Progress: {i}/{total}")

            source_tensor_bytes = tensor_nbytes(v)
            if (
                k.endswith(".weight")
                and v.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and v.ndim == 2
            ):
                matrix_source_bytes += source_tensor_bytes

            matched_profile_rule = next((name for name in blacklist if name in k), None)
            if matched_profile_rule is not None:
                new_sd[k], count_name = preserve_tensor(v, source_kind)
                counts[count_name] += 1
                coverage_bytes["kept by profile"] += source_tensor_bytes
                profile_keep_bytes[matched_profile_rule] += source_tensor_bytes
                continue

            matched_global_rule = next((name for name in protected_substrings if name in k), None)
            if matched_global_rule is not None:
                new_sd[k], count_name = preserve_tensor(v, source_kind)
                counts[count_name] += 1
                coverage_bytes["kept by global protection"] += source_tensor_bytes
                global_keep_bytes[matched_global_rule] += source_tensor_bytes
                continue

            basic_candidate = is_quantizable_weight(k, v, protected_substrings=())
            if target_format in ("fp8", "int8"):
                candidate = basic_candidate
            elif convrot_target:
                candidate = basic_candidate
            else:
                candidate = can_quantize_weight(
                    k, v, protected_substrings=(), alignment=quant_alignment
                )

            if candidate:
                base_k_file = k.replace(".weight", "")
                base_k_meta = base_k_file

                # Current Forge kitchen kernels accept FP16/BF16 input only.  Keeping
                # this tensor in BF16 also avoids creating a large FP32 copy per layer.
                v_tensor = v.to(device=device, dtype=torch.bfloat16)

                if target_format == "fp8" or (fp8_layers and any(name in k for name in fp8_layers)):
                    log(f"FP8: {k}")
                    weight_scale = (v_tensor.abs().max() / 448.0).clamp(min=1e-12).float()
                    weight_quantized = ck.quantize_per_tensor_fp8(v_tensor, weight_scale)
                    stored_weight = weight_quantized.cpu()
                    stored_scale = weight_scale.to(torch.bfloat16).cpu()
                    layer_conf = {"format": "float8_e4m3fn"}
                    comfy_tensor = encode_quant_config(layer_conf)
                    new_sd[k] = stored_weight
                    new_sd[f"{base_k_file}.weight_scale"] = stored_scale
                    new_sd[f"{base_k_file}.comfy_quant"] = comfy_tensor
                    quant_map["layers"][base_k_meta] = layer_conf
                    counts["fp8"] += 1
                    coverage_bytes["fp8"] += source_tensor_bytes
                    coverage_bytes["fp8 stored"] += (
                        tensor_nbytes(stored_weight)
                        + tensor_nbytes(stored_scale)
                        + tensor_nbytes(comfy_tensor)
                    )
                    if device == "cuda":
                        del v_tensor
                    continue

                requested_int8_convrot = target_format == "int8_convrot"
                requested_int4_convrot = target_format == "int4_convrot"
                requested_w4a8_convrot = target_format == "w4a8_convrot"
                layer_format = target_format
                layer_gs = best_convrot_groupsize(v.size(1)) if convrot_target else None

                # ConvRot rotates K (in_features), not N.  The previous converter
                # required both dimensions to satisfy one global alignment, which
                # left many otherwise valid layers in BF16.  Follow the current
                # comfy-model-tools policy more closely: pick the largest supported
                # ConvRot group per layer and only apply format-specific constraints.
                if requested_int8_convrot:
                    if layer_gs is None or v.size(0) < 8:
                        layer_format = None
                elif requested_int4_convrot:
                    if layer_gs is None or v.size(0) < 8:
                        layer_format = None
                    elif v.size(1) % INT4_QUANT_GROUPSIZE != 0:
                        layer_format = "int8_convrot_fallback"
                elif requested_w4a8_convrot:
                    # Current Comfy model tools use W4A8 when K is divisible by
                    # 256 and N>=64, otherwise they fall back to INT8 ConvRot.
                    if v.size(1) % CONVROT_GROUPSIZE != 0 or v.size(0) < 64:
                        layer_format = "int8_convrot_fallback" if layer_gs is not None and v.size(0) >= 8 else None

                if layer_format is None:
                    new_sd[k], count_name = preserve_tensor(v, source_kind)
                    counts[count_name] += 1
                    reason_counts["convrot shape kept"] += 1
                    coverage_bytes["convrot shape kept"] += source_tensor_bytes
                    del v_tensor
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    continue

                def quantize_as_int8_convrot(ready_tensor, group_size):
                    return TensorWiseINT8Layout.quantize(
                        ready_tensor,
                        is_weight=True,
                        per_channel=True,
                        convrot=True,
                        convrot_groupsize=group_size,
                    )

                if layer_format in ("int8", "int8_convrot", "int8_convrot_fallback"):
                    layout = TensorWiseINT8Layout
                    fmt_name = "int8_tensorwise"
                elif layer_format == "int4_convrot":
                    layout = TensorCoreConvRotW4A4Layout
                    fmt_name = "convrot_w4a4"
                elif layer_format == "w4a8_convrot":
                    layout = AsymW4A8Int8Layout
                    fmt_name = "asym_w4a8_int8"
                elif target_format == "mxfp8":
                    layout = TensorCoreMXFP8Layout
                    fmt_name = "mxfp8"
                else:
                    layout = TensorCoreNVFP4Layout
                    fmt_name = "nvfp4"

                log_label = layer_format.upper().replace("_FALLBACK", " FALLBACK")
                log(f"{log_label}: {k}" + (f" (gs={layer_gs})" if layer_gs else ""))

                qdata = params = tensors = v_tensor_ready = None
                used_format = layer_format
                try:
                    # Do not cast to float32 here: recent Forge kernels reject it
                    # ("Unsupported dtype code: 0") and the temporary FP32 copy can
                    # exhaust VRAM on large DiTs.
                    v_tensor_ready = v_tensor.contiguous()
                    if layer_format in ("int8_convrot", "int8_convrot_fallback"):
                        qdata, params = quantize_as_int8_convrot(v_tensor_ready, layer_gs)
                    elif layer_format == "int4_convrot":
                        qdata, params = layout.quantize(
                            v_tensor_ready,
                            convrot_groupsize=layer_gs,
                            quant_group_size=INT4_QUANT_GROUPSIZE,
                        )
                    elif layer_format == "w4a8_convrot":
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
                except Exception as primary_error:
                    # If a 4-bit backend rejects a layer, use the proven INT8
                    # ConvRot path instead of silently bloating the output with BF16.
                    if layer_format in ("int4_convrot", "w4a8_convrot") and layer_gs is not None and v.size(0) >= 8:
                        log(
                            f"Warning: {layer_format} failed for {k}: {primary_error}; "
                            f"trying INT8 ConvRot fallback (gs={layer_gs})"
                        )
                        try:
                            layout = TensorWiseINT8Layout
                            fmt_name = "int8_tensorwise"
                            qdata, params = quantize_as_int8_convrot(v_tensor_ready, layer_gs)
                            used_format = "int8_convrot_fallback"
                        except Exception as fallback_error:
                            log(f"Warning: INT8 ConvRot fallback also failed for {k}: {fallback_error}")
                            new_sd[k], count_name = preserve_tensor(v, source_kind)
                            counts[count_name] += 1
                            reason_counts["quantization failed kept"] += 1
                            coverage_bytes["quantization failed kept"] += source_tensor_bytes
                            qdata = params = None
                    else:
                        log(f"Warning: quantization failed for {k}: {primary_error}")
                        new_sd[k], count_name = preserve_tensor(v, source_kind)
                        counts[count_name] += 1
                        reason_counts["quantization failed kept"] += 1
                        coverage_bytes["quantization failed kept"] += source_tensor_bytes
                        qdata = params = None

                if qdata is not None:
                    tensors = layout.state_dict_tensors(qdata, params)
                    stored_bytes = 0
                    for suffix, tensor in tensors.items():
                        out_key = f"{base_k_file}.weight{suffix}"
                        if FLOAT8_E8M0 is not None and tensor.dtype == FLOAT8_E8M0:
                            stored = tensor.view(torch.uint8).cpu()
                        elif tensor.dtype in FP8_DTYPES:
                            stored = tensor.view(torch.uint8).cpu().view(tensor.dtype)
                        else:
                            stored = tensor.cpu()
                        new_sd[out_key] = stored
                        stored_bytes += tensor_nbytes(stored)

                    layer_conf = {"format": fmt_name}
                    if used_format in ("int8_convrot", "int8_convrot_fallback"):
                        layer_conf["convrot"] = True
                        layer_conf["convrot_groupsize"] = layer_gs
                    elif used_format == "int4_convrot":
                        layer_conf["convrot_groupsize"] = layer_gs
                        layer_conf["quant_group_size"] = INT4_QUANT_GROUPSIZE
                    elif used_format == "w4a8_convrot":
                        layer_conf["group_size"] = W4A8_QUANT_GROUPSIZE
                        layer_conf["convrot_groupsize"] = CONVROT_GROUPSIZE

                    comfy_tensor = encode_quant_config(layer_conf)
                    new_sd[f"{base_k_file}.comfy_quant"] = comfy_tensor
                    stored_bytes += tensor_nbytes(comfy_tensor)
                    quant_map["layers"][base_k_meta] = layer_conf
                    count_key = "int8_convrot fallback" if used_format == "int8_convrot_fallback" else used_format
                    counts[count_key] += 1
                    coverage_bytes[count_key] += source_tensor_bytes
                    coverage_bytes[f"{count_key} stored"] += stored_bytes

                # Explicitly drop all CUDA temporaries before the next layer.  The
                # quantization layouts can allocate several working buffers, so
                # retaining one iteration is enough to OOM a 16 GB GPU.
                del qdata, params, tensors, v_tensor_ready, v_tensor
                if device == "cuda":
                    torch.cuda.empty_cache()
            else:
                new_sd[k], count_name = preserve_tensor(v, source_kind)
                counts[count_name] += 1
                coverage_bytes["kept non-candidate"] += source_tensor_bytes

    final_metadata = OrderedDict(temp_diffusers_meta)
    if quant_map["layers"]:
        final_metadata["_quantization_metadata"] = json.dumps(quant_map)
        first_quant_layer = next(iter(quant_map["layers"]))
        log(f"Quantization metadata: {len(quant_map['layers'])} layers, first key: {first_quant_layer}")
    final_metadata["converted_by"] = "Star Ultimate Model Converter"

    log(f"Saving safely | Type: {active_model_type} | Path: {output_path}")
    save_safetensors_atomic(new_sd, output_path, final_metadata, log=log)

    output_bytes = os.path.getsize(output_path)
    duration = time.time() - start_time
    reduction = (1 - output_bytes / input_bytes) * 100 if input_bytes else 0
    layers_desc = ", ".join(f"{n} {name}" for name, n in counts.most_common())
    if target_format not in ("fp16", "fp32"):
        quantized_coverage_names = tuple(
            dict.fromkeys((target_format, "fp8", "int8_convrot fallback"))
        )
        coverage_parts = []
        for name in (
            *quantized_coverage_names,
            "convrot shape kept",
            "quantization failed kept",
            "kept by profile",
            "kept by global protection",
            "kept non-candidate",
        ):
            num_bytes = coverage_bytes.get(name, 0)
            if num_bytes:
                coverage_parts.append(f"{name}: {format_size(num_bytes)} source")
        if coverage_parts:
            log("Quantization coverage | " + " | ".join(coverage_parts))
        storage_parts = []
        for name in quantized_coverage_names:
            source_num = coverage_bytes.get(name, 0)
            stored_num = coverage_bytes.get(f"{name} stored", 0)
            if source_num and stored_num:
                storage_parts.append(f"{name}: {format_size(source_num)} -> {format_size(stored_num)}")
        if storage_parts:
            log("Quantized storage | " + " | ".join(storage_parts))
        if profile_keep_bytes:
            breakdown = " | ".join(
                f"{name}: {format_size(num_bytes)}"
                for name, num_bytes in profile_keep_bytes.most_common(8)
            )
            log(f"Kept by profile breakdown | {breakdown}")
        if global_keep_bytes:
            breakdown = " | ".join(
                f"{name}: {format_size(num_bytes)}"
                for name, num_bytes in global_keep_bytes.most_common(8)
            )
            log(f"Kept by global protection breakdown | {breakdown}")

        quantized_source_bytes = sum(
            coverage_bytes.get(name, 0)
            for name in quantized_coverage_names
        )
        if matrix_source_bytes:
            matrix_coverage = 100.0 * quantized_source_bytes / matrix_source_bytes
            log(
                f"Matrix quantization coverage | {format_size(quantized_source_bytes)} / "
                f"{format_size(matrix_source_bytes)} = {matrix_coverage:.1f}%"
            )
            if matrix_coverage < 70.0:
                log(
                    "WARNING: Less than 70% of 2D floating-point weight bytes were quantized. "
                    "Check the profile/global-protection breakdown before trusting the output size."
                )
        if reason_counts:
            reasons = ", ".join(f"{n} {name}" for name, n in reason_counts.most_common())
            log(f"ConvRot fallbacks/kept: {reasons}")
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
