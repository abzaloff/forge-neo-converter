# Civitai description draft

## Forge Neo Krea-2 Converter & LoRA Tools

A community extension for Stable Diffusion Forge Neo, based on and extending the original [Forge Neo Converter by abzaloff](https://github.com/abzaloff/forge-neo-converter).

It adds a RAM-friendlier streaming conversion path for supported large Krea-2 models and integrates Krea2 LoRA analysis and stripping tools.

### What it does

- Convert supported Krea-2 `.safetensors` models to Forge Neo-compatible quantized formats, including `w4a8_convrot` and `int4_convrot` where supported by the installed Forge Neo/comfy-kitchen build.
- Process supported high-precision input tensors one at a time to reduce peak system RAM pressure during large-model conversion.
- Analyze Krea2 LoRAs before modifying them: signature, dtypes, tensor groups, tensor counts and byte distribution.
- Strip Krea2 LoRAs using several profiles: **Max**, **Compact 25%**, **Balanced 50%**, and **Light**.
- Report kept-byte ratio and flag aggressive stripping that may increase fidelity risk.

### Tested example: large Krea-2 model

On a system with **16 GB system RAM** and **RTX 2060 6 GB VRAM**:

- 24.48 GB BF16 Krea-2 → **7.15 GB `w4a8_convrot`** — successful CUDA conversion.
- 24.48 GB BF16 Krea-2 → **6.43 GB `int4_convrot`** — successful CUDA conversion.

These are real test results, not a guarantee for every model or hardware/software configuration.

### Tested example: Krea2 LoRA

`Krea2Y3nnefer.safetensors` (218 MB) was tested with:

- **Balanced 50%:** 115.6 MB
- **Compact 25%:** 64.4 MB
- **Max / txtfusion-only:** 13.2 MB

Character/appearance LoRAs may remain useful after stripping, while concept/style LoRAs can lose significant visual behavior. Keep the original and A/B test before relying on a stripped LoRA.

### Performance example

With the same prompt, LoRA weight, model, 8 steps, Euler, CFG 1 and seed, the tested Y3nnefer LoRA produced these approximate generation times on the same system:

- Light/original: 65.1 s
- Balanced 50%: 59.2 s
- Compact 25%: 45.7 s
- Max: 35.3 s

Results depend on the exact backend, model, hardware and Forge Neo version.

### Installation

Install/copy the extension into the Forge Neo `extensions` directory and restart Forge Neo completely. See the GitHub README for details.

### Credits

This project is a community/derivative extension and is not an official release by the upstream authors.

- **Forge Neo Converter base:** [abzaloff/forge-neo-converter](https://github.com/abzaloff/forge-neo-converter)
- **Krea2 LoRA stripping source:** [Winnougan/Krea2_LoRA_Stripper](https://github.com/Winnougan/Krea2_LoRA_Stripper)
- **Original conversion/profile upstream:** [Starnodes Model Converter](https://github.com/Starnodes2024/comfyui-starnodes-modelconverter)
- **Community integration, fixes, tests and Forge Neo UI:** `dbodik-git`

The Krea2 LoRA stripping integration is derived from Winnougan's `batch_strip_krea2.py` workflow, including its Krea2 signature detection, strip-prefix approach and fidelity-risk heuristic.

### Disclaimer

Experimental/community software. Back up models and LoRAs before conversion or stripping. Respect the licenses and redistribution terms of all upstream projects and the models/LoRAs you process.