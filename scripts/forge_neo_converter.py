import os
import sys
import traceback

import gradio as gr

from modules import call_queue, paths, script_callbacks, shared, ui
from modules.ui_components import ToolButton


EXTENSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXTENSION_ROOT not in sys.path:
    sys.path.insert(0, EXTENSION_ROOT)

from forge_neo_converter_core import TARGET_FORMATS, convert_model, model_types  # noqa: E402


MODEL_EXTENSIONS = (".safetensors",)
_MODEL_CHOICES = {}


def _candidate_model_dirs():
    dirs = [
        os.path.join(paths.models_path, "diffusion_models"),
        os.path.join(paths.models_path, "unet"),
        os.path.join(paths.models_path, "Stable-diffusion"),
    ]
    for extra_dir in getattr(shared.cmd_opts, "ckpt_dirs", []) or []:
        dirs.append(extra_dir)

    result = []
    seen = set()
    for directory in dirs:
        if not directory:
            continue
        full = os.path.abspath(directory)
        key = os.path.normcase(full)
        if key in seen or not os.path.isdir(full):
            continue
        seen.add(key)
        result.append(full)
    return result


def _short_name(path, roots):
    abs_path = os.path.abspath(path)
    for root in roots:
        try:
            rel = os.path.relpath(abs_path, root)
        except ValueError:
            continue
        if not rel.startswith(".."):
            return rel.replace("\\", "/")
    return abs_path


def list_model_choices():
    roots = _candidate_model_dirs()
    found = []
    for root in roots:
        for current_root, _dirs, files in os.walk(root):
            for filename in files:
                if filename.lower().endswith(MODEL_EXTENSIONS):
                    found.append(os.path.join(current_root, filename))

    labels = {}
    for path in sorted(set(found), key=lambda p: p.lower()):
        label = _short_name(path, roots)
        if label in labels:
            label = path
        labels[label] = path

    _MODEL_CHOICES.clear()
    _MODEL_CHOICES.update(labels)
    return list(labels.keys())


def resolve_model_path(model_name):
    if not _MODEL_CHOICES:
        list_model_choices()
    return _MODEL_CHOICES.get(model_name, model_name)


def run_conversion(model_name, model_type, target_format, device):
    def log(message):
        print(f"[Forge Neo Converter] {message}")

    try:
        model_path = resolve_model_path(model_name)
        status, _ = convert_model(model_path, model_type, target_format, device, log=log)
        return status
    except Exception as e:
        log(f"Error: {e}")
        details = traceback.format_exc()
        log(details)
        return f"Error: {e}\n\n{details}"


def refreshed_model_args():
    choices = list(_MODEL_CHOICES.keys())
    return {"choices": choices, "value": choices[0] if choices else None}


def refresh_model_dropdown():
    list_model_choices()
    return gr.update(**refreshed_model_args())


def create_converter_tab():
    choices = list_model_choices()
    types = model_types()
    default_type = "Flux1 / Flux2" if "Flux1 / Flux2" in types else (types[0] if types else None)

    with gr.Blocks(analytics_enabled=False) as converter_interface:
        with gr.Row(equal_height=False):
            with gr.Column(variant="compact", scale=2):
                with gr.Row():
                    model_name = gr.Dropdown(
                        label="Model",
                        choices=choices,
                        value=choices[0] if choices else None,
                        elem_id="forge_neo_converter_model_name",
                        scale=8,
                    )
                    refresh_models = ToolButton(
                        value=ui.refresh_symbol,
                        elem_id="forge_neo_converter_refresh_models",
                        tooltip="Refresh model list",
                        scale=1,
                    )

                model_type = gr.Dropdown(
                    label="Model type",
                    choices=types,
                    value=default_type,
                    elem_id="forge_neo_converter_model_type",
                )
                target_format = gr.Dropdown(
                    label="Target format",
                    choices=TARGET_FORMATS,
                    value="nvfp4",
                    elem_id="forge_neo_converter_target_format",
                )
                device = gr.Radio(
                    label="Device",
                    choices=["cuda", "cpu"],
                    value="cuda",
                    elem_id="forge_neo_converter_device",
                )
                convert_button = gr.Button("Convert", variant="primary", elem_id="forge_neo_converter_convert")

            with gr.Column(variant="panel", scale=3):
                log_output = gr.Textbox(
                    label="Log",
                    value="Ready.",
                    lines=24,
                    max_lines=40,
                    elem_id="forge_neo_converter_log",
                )

        for comp in (model_name, model_type, target_format, device, convert_button, log_output, refresh_models):
            comp.do_not_save_to_config = True

        refresh_models.click(
            fn=refresh_model_dropdown,
            outputs=[model_name],
            queue=False,
            show_progress=False,
        )

        convert_button.click(
            fn=lambda: "Starting conversion...",
            outputs=[log_output],
            queue=False,
            show_progress=False,
        ).then(
            fn=call_queue.wrap_queued_call(run_conversion),
            inputs=[model_name, model_type, target_format, device],
            outputs=[log_output],
            show_progress=True,
        )

    return [(converter_interface, "Converter", "forge_neo_converter_tab")]


script_callbacks.on_ui_tabs(create_converter_tab, name="forge_neo_converter_tab")
