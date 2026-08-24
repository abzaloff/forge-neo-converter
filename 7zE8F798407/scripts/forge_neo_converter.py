import importlib
import os
import sys
import traceback

import gradio as gr
import torch

from modules import call_queue, paths, script_callbacks, shared, ui
from modules.ui_components import ToolButton


EXTENSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXTENSION_ROOT not in sys.path:
    sys.path.insert(0, EXTENSION_ROOT)

import forge_neo_converter_core as converter_core  # noqa: E402


# Forge can reload extension scripts without clearing their imported modules.
# Refresh an older cached core so newly added converter modes are available.
if not all(
    hasattr(converter_core, name)
    for name in ("TEXT_ENCODER_PROFILE", "convert_text_encoder")
) or "w4a8_convrot" not in converter_core.TARGET_FORMATS:
    converter_core = importlib.reload(converter_core)

TARGET_FORMATS = converter_core.TARGET_FORMATS
TEXT_ENCODER_PROFILE = converter_core.TEXT_ENCODER_PROFILE
convert_model = converter_core.convert_model
convert_text_encoder = converter_core.convert_text_encoder
model_types = converter_core.model_types


MODEL_EXTENSIONS = (".safetensors",)
_MODEL_CHOICES = {}
_TEXT_ENCODER_CHOICES = {}


def _existing_unique_dirs(dirs):
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


def _candidate_model_dirs():
    dirs = [
        os.path.join(paths.models_path, "diffusion_models"),
        os.path.join(paths.models_path, "unet"),
        os.path.join(paths.models_path, "Stable-diffusion"),
        *(getattr(shared.cmd_opts, "ckpt_dirs", []) or []),
    ]
    return _existing_unique_dirs(dirs)


def _candidate_text_encoder_dirs():
    dirs = [
        os.path.join(paths.models_path, "text_encoder"),
        os.path.join(paths.models_path, "text_encoders"),
        os.path.join(paths.models_path, "clip"),
        *(getattr(shared.cmd_opts, "text_encoder_dirs", []) or []),
    ]
    return _existing_unique_dirs(dirs)


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


def _list_file_choices(roots, destination):
    found = []
    visited_dirs = set()
    for root in roots:
        for current_root, dirs, files in os.walk(root, followlinks=True):
            real_root = os.path.normcase(os.path.realpath(current_root))
            if real_root in visited_dirs:
                dirs[:] = []
                continue
            visited_dirs.add(real_root)

            for filename in files:
                if filename.lower().endswith(MODEL_EXTENSIONS):
                    found.append(os.path.join(current_root, filename))

    labels = {}
    for path in sorted(set(found), key=lambda p: p.lower()):
        label = _short_name(path, roots)
        if label in labels:
            label = path
        labels[label] = path

    ordered_labels = dict(sorted(labels.items(), key=lambda item: item[0].lower()))
    destination.clear()
    destination.update(ordered_labels)
    return list(ordered_labels.keys())


def list_model_choices():
    return _list_file_choices(_candidate_model_dirs(), _MODEL_CHOICES)


def list_text_encoder_choices():
    return _list_file_choices(_candidate_text_encoder_dirs(), _TEXT_ENCODER_CHOICES)


def resolve_model_path(model_name):
    if not _MODEL_CHOICES:
        list_model_choices()
    return _MODEL_CHOICES.get(model_name, model_name)


def resolve_text_encoder_path(text_encoder_name):
    if not _TEXT_ENCODER_CHOICES:
        list_text_encoder_choices()
    return _TEXT_ENCODER_CHOICES.get(text_encoder_name, text_encoder_name)


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
    finally:
        # The converter uses large temporary CUDA buffers.  Forge otherwise keeps
        # them cached after this queued job, which can make the next generation OOM.
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_text_encoder_conversion(text_encoder_name, target_format, device):
    def log(message):
        print(f"[Forge Neo Converter] {message}")

    try:
        text_encoder_path = resolve_text_encoder_path(text_encoder_name)
        status, _ = convert_text_encoder(text_encoder_path, target_format, device, log=log)
        return status
    except Exception as e:
        log(f"Error: {e}")
        details = traceback.format_exc()
        log(details)
        return f"Error: {e}\n\n{details}"
    finally:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def refreshed_model_args():
    choices = list(_MODEL_CHOICES.keys())
    return {"choices": choices, "value": choices[0] if choices else None}


def refresh_model_dropdown():
    list_model_choices()
    return gr.update(**refreshed_model_args())


def refreshed_text_encoder_args():
    choices = list(_TEXT_ENCODER_CHOICES.keys())
    return {"choices": choices, "value": choices[0] if choices else None}


def refresh_text_encoder_dropdown():
    list_text_encoder_choices()
    return gr.update(**refreshed_text_encoder_args())


def create_converter_tab():
    model_choices = list_model_choices()
    text_encoder_choices = list_text_encoder_choices()
    types = [model_type for model_type in model_types() if model_type != TEXT_ENCODER_PROFILE]
    default_type = "Flux1 / Flux2" if "Flux1 / Flux2" in types else (types[0] if types else None)

    with gr.Blocks(analytics_enabled=False) as converter_interface:
        with gr.Tabs(elem_id="forge_neo_converter_modes"):
            with gr.Tab("Model mode", id="model", elem_id="forge_neo_converter_model_mode"):
                with gr.Row(equal_height=False):
                    with gr.Column(variant="compact", scale=2):
                        with gr.Row():
                            model_name = gr.Dropdown(
                                label="Model",
                                choices=model_choices,
                                value=model_choices[0] if model_choices else None,
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
                        model_target_format = gr.Dropdown(
                            label="Target format",
                            choices=TARGET_FORMATS,
                            value="nvfp4",
                            elem_id="forge_neo_converter_model_target_format",
                        )
                        model_device = gr.Radio(
                            label="Device",
                            choices=["cuda", "cpu"],
                            value="cuda",
                            elem_id="forge_neo_converter_model_device",
                        )
                        model_convert_button = gr.Button(
                            "Convert model",
                            variant="primary",
                            elem_id="forge_neo_converter_convert_model",
                        )

                    with gr.Column(variant="panel", scale=3):
                        model_log_output = gr.Textbox(
                            label="Log",
                            value="Ready.",
                            lines=24,
                            max_lines=40,
                            elem_id="forge_neo_converter_model_log",
                        )

            with gr.Tab("Text encoder mode", id="text_encoder", elem_id="forge_neo_converter_text_encoder_mode"):
                with gr.Row(equal_height=False):
                    with gr.Column(variant="compact", scale=2):
                        with gr.Row():
                            text_encoder_name = gr.Dropdown(
                                label="Text encoder",
                                choices=text_encoder_choices,
                                value=text_encoder_choices[0] if text_encoder_choices else None,
                                elem_id="forge_neo_converter_text_encoder_name",
                                scale=8,
                            )
                            refresh_text_encoders = ToolButton(
                                value=ui.refresh_symbol,
                                elem_id="forge_neo_converter_refresh_text_encoders",
                                tooltip="Refresh text encoder list",
                                scale=1,
                            )

                        text_encoder_target_format = gr.Dropdown(
                            label="Target format",
                            choices=TARGET_FORMATS,
                            value="nvfp4",
                            elem_id="forge_neo_converter_text_encoder_target_format",
                        )
                        text_encoder_device = gr.Radio(
                            label="Device",
                            choices=["cuda", "cpu"],
                            value="cuda",
                            elem_id="forge_neo_converter_text_encoder_device",
                        )
                        text_encoder_convert_button = gr.Button(
                            "Convert text encoder",
                            variant="primary",
                            elem_id="forge_neo_converter_convert_text_encoder",
                        )

                    with gr.Column(variant="panel", scale=3):
                        text_encoder_log_output = gr.Textbox(
                            label="Log",
                            value="Ready. The Text-Encoder profile will be used automatically.",
                            lines=24,
                            max_lines=40,
                            elem_id="forge_neo_converter_text_encoder_log",
                        )

        components = (
            model_name,
            model_type,
            model_target_format,
            model_device,
            model_convert_button,
            model_log_output,
            refresh_models,
            text_encoder_name,
            text_encoder_target_format,
            text_encoder_device,
            text_encoder_convert_button,
            text_encoder_log_output,
            refresh_text_encoders,
        )
        for comp in components:
            comp.do_not_save_to_config = True

        refresh_models.click(
            fn=refresh_model_dropdown,
            outputs=[model_name],
            queue=False,
            show_progress=False,
        )

        refresh_text_encoders.click(
            fn=refresh_text_encoder_dropdown,
            outputs=[text_encoder_name],
            queue=False,
            show_progress=False,
        )

        model_convert_button.click(
            fn=lambda: "Starting model conversion...",
            outputs=[model_log_output],
            queue=False,
            show_progress=False,
        ).then(
            fn=call_queue.wrap_queued_call(run_conversion),
            inputs=[model_name, model_type, model_target_format, model_device],
            outputs=[model_log_output],
            show_progress=True,
        )

        text_encoder_convert_button.click(
            fn=lambda: "Starting text encoder conversion...",
            outputs=[text_encoder_log_output],
            queue=False,
            show_progress=False,
        ).then(
            fn=call_queue.wrap_queued_call(run_text_encoder_conversion),
            inputs=[text_encoder_name, text_encoder_target_format, text_encoder_device],
            outputs=[text_encoder_log_output],
            show_progress=True,
        )

    return [(converter_interface, "Converter", "forge_neo_converter_tab")]


script_callbacks.on_ui_tabs(create_converter_tab, name="forge_neo_converter_tab")
