# Attribution

This repository is a community/derivative extension of multiple upstream projects.

## Forge Neo Converter base

- **Project:** [abzaloff/forge-neo-converter](https://github.com/abzaloff/forge-neo-converter)
- **Role:** Base Forge Neo converter integration and project structure.

## Krea2 LoRA stripping

- **Project:** [Winnougan/Krea2_LoRA_Stripper](https://github.com/Winnougan/Krea2_LoRA_Stripper)
- **Role:** Original Krea2 LoRA stripping workflow.
- **Integrated functionality:** Krea2 signature detection, DIT/UNET strip-prefix logic, tensor-byte accounting, output generation, and the kept-byte fidelity-risk heuristic are derived from the upstream `batch_strip_krea2.py` workflow.

The current Forge Neo extension integrates that workflow into the Forge UI and adds the Analyzer plus additional `Max`, `Compact 25%`, `Balanced 50%`, and `Light` profiles.

## Conversion upstream

- **Project:** [Starnodes Model Converter](https://github.com/Starnodes2024/comfyui-starnodes-modelconverter)
- **Role:** Upstream conversion workflow/profile data used by the original Forge Neo converter project.

## Community changes

The `dbodik-git` repository adds community modifications, including streaming input handling for supported large safetensors, Krea2 LoRA UI integration, analysis tools, additional stripping profiles, fixes, tests, and documentation.

Please consult each upstream repository for its current license and redistribution terms before distributing derivative copies.