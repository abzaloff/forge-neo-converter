# Forge Neo Converter

Forge Neo Converter is a Forge Neo extension for converting `.safetensors` diffusion model files between precision and quantization formats directly from the Forge UI.

It adds a top-level **Converter** tab with controls for:

- model file
- model architecture profile
- target format
- conversion device
- conversion log output

Converted models are saved next to the selected source model. The output name follows the same naming behavior as the source converter: an existing precision suffix is stripped when detected, then the selected target format is appended.

Example:

```text
ZIT_Luxury_1_0_bf16.safetensors
ZIT_Luxury_1_0-int8_convrot.safetensors
```

## Source and Attribution

This extension is based on the ComfyUI custom node project:

[Starnodes Model Converter](https://github.com/Starnodes2024/comfyui-starnodes-modelconverter)

The conversion logic, model profiles, quantization formats, layer preservation profiles, file naming behavior, and metadata handling are adapted from the original **Star Ultimate Model Converter** node.

This Forge Neo extension ports that workflow into a native Forge Neo tab and adjusts quantization metadata so converted models load correctly through Forge Neo's mixed precision loader.

## Features

- Top-level **Converter** tab in Forge Neo.
- Converts `.safetensors` model files in place.
- Saves converted models in the same directory as the selected source model.
- Supports architecture-specific model profiles from the original Starnodes converter.
- Supports Forge Neo-compatible mixed precision metadata.
- Shows a text log for conversion progress, layer statistics, output path, and errors.
- Scans standard Forge model folders and configured checkpoint directories.

## Supported Target Formats

- `nvfp4`
- `fp8`
- `mxfp8`
- `int8`
- `int8_convrot`
- `int4_convrot` — ConvRot W4A4 quantization with 4-bit weights and group-wise scales.
- `fp16`
- `fp32`

Quantized formats require `comfy-kitchen`. `int4_convrot` additionally requires a current Forge Neo/comfy-kitchen build that provides the ConvRot W4A4 layout.

## Model Search Paths

The model dropdown scans `.safetensors` files from:

```text
models/diffusion_models
models/unet
models/Stable-diffusion
```

It also scans directories passed to Forge Neo with:

```text
--ckpt-dir
```

If models are added while Forge Neo is already running, click **Refresh** in the Converter tab.

## Installation

Clone or copy this repository into your Forge Neo `extensions` directory:

```bash
cd /path/to/sd-webui-forge-neo/extensions
git clone <this-repository-url> forge-neo-converter
```

Install dependencies in the Forge Neo environment if they are not already installed:

```bash
pip install -r requirements.txt
```

If your Forge Neo installation uses `uv`, install the dependencies through the same environment used by your `webui-user.bat` or launch setup.

Restart Forge Neo completely after installation.

## Usage

1. Start Forge Neo.
2. Open the **Converter** tab.
3. Select a `.safetensors` model from the dropdown.
4. Select the matching model type/profile.
5. Select the target format.
6. Select `cuda` or `cpu`.
7. Click **Convert**.
8. Watch the log until the conversion completes.

The converted file is written to the same folder as the source model.

## Forge Neo Loading Notes

For quantized output, Forge Neo should detect the model as mixed precision. During model loading, a healthy log usually includes a line similar to:

```text
Using MixedPrecision for Model
```

If Forge Neo prints unexpected `weight_scale` keys, for example:

```text
NextDiT Unexpected: [...weight_scale...]
```

then the file was not created with Forge-compatible quantization metadata. Delete that converted file, fully restart Forge Neo so the latest extension code is loaded, and convert again.

During conversion, the log should include a metadata line like:

```text
Quantization metadata: 90 layers, first key: model.diffusion_model.layers.0.feed_forward.w1
```

The exact layer count depends on the model and selected profile.

## Requirements

- Forge Neo
- Python environment used by Forge Neo
- PyTorch
- safetensors
- comfy-kitchen
- NVIDIA GPU recommended for CUDA conversion and hardware-friendly quantized formats

See [requirements.txt](requirements.txt).

## Upstream Profile Data

The model profile data used by this extension is kept in [models.json](models.json). It is adapted from the upstream Starnodes Model Converter profile definitions.

Only the converter workflow is exposed in Forge Neo. The additional ComfyUI-only node UI and optional AIO splitter controls are not exposed in the Forge Neo tab.

## License

This extension is a Forge Neo port based on the Starnodes Model Converter project. Respect the license and terms of the upstream project and the licenses of any models you convert.
