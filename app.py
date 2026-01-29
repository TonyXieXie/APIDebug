import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

CONFIG_PATH = os.environ.get("LLM_DEBUG_CONFIG", "configs.json")
DEFAULT_TIMEOUT = 60
OPENAI_STANDARD_MODEL = "gpt-4o-mini"
OPENAI_STANDARD_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a haiku about debugging API requests."},
]


def _flash(kind: str, message: str) -> None:
    st.session_state["_flash"] = {"kind": kind, "message": message}


def _show_flash(location: Any) -> None:
    payload = st.session_state.pop("_flash", None)
    if not payload:
        return
    kind = payload.get("kind", "info")
    message = payload.get("message", "")
    if not message:
        return
    fn = getattr(location, kind, None)
    if callable(fn):
        fn(message)
    else:
        location.info(message)


def load_configs(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        st.error(f"Config file not found: {path}")
        st.stop()
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON in config file: {exc}")
        st.stop()
    if not isinstance(data, list):
        st.error("Config file must contain a JSON list of configs.")
        st.stop()
    return data


def validate_configs(data: Any) -> List[Dict[str, Any]]:
    if not isinstance(data, list):
        raise ValueError("Config JSON must be a list.")
    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Config entry #{index + 1} must be an object.")
        normalized.append(item)
    return normalized


def resolve_target_path(base_dir: Path, raw_path: str) -> Path:
    if not raw_path.strip():
        raise ValueError("Path is empty.")
    candidate = Path(raw_path.strip())
    if candidate.is_absolute():
        raise ValueError("Only relative paths are allowed.")
    target = (base_dir / candidate).resolve()
    if base_dir != target and base_dir not in target.parents:
        raise ValueError("Invalid path (must stay within the app folder).")
    return target


def write_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def backup_file(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".bak.{timestamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def apply_provider_overrides(
    config: Dict[str, Any],
    base_url_override: str,
    endpoint_override: str,
    model_override: str,
) -> Dict[str, Any]:
    updated = dict(config)
    if base_url_override.strip():
        updated["base_url"] = base_url_override.strip()
    if endpoint_override.strip():
        updated["endpoint"] = endpoint_override.strip()
    if model_override.strip():
        updated["model"] = model_override.strip()
    return updated


def parse_extra_params(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid extra params JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Extra params JSON must be an object.")
    return data


def apply_parameter_overrides(
    config: Dict[str, Any],
    *,
    api_mode: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    timeout_seconds: int,
    extra_params: Dict[str, Any],
) -> Dict[str, Any]:
    updated = dict(config)

    default_params = updated.get("default_params") or {}
    if not isinstance(default_params, dict):
        default_params = {}

    default_params = dict(default_params)
    default_params["temperature"] = float(temperature)
    default_params["top_p"] = float(top_p)

    if api_mode == "responses":
        default_params.pop("frequency_penalty", None)
        default_params.pop("presence_penalty", None)
        default_params.pop("max_tokens", None)
        if int(max_tokens) > 0:
            default_params["max_output_tokens"] = int(max_tokens)
        else:
            default_params.pop("max_output_tokens", None)
    else:
        default_params.pop("max_output_tokens", None)
        default_params["frequency_penalty"] = float(frequency_penalty)
        default_params["presence_penalty"] = float(presence_penalty)
        if int(max_tokens) > 0:
            default_params["max_tokens"] = int(max_tokens)
        else:
            default_params.pop("max_tokens", None)

    updated["default_params"] = default_params
    updated["timeout_seconds"] = int(timeout_seconds)
    updated["extra_params"] = dict(extra_params)
    return updated


def build_url(base_url: str, endpoint: str) -> str:
    endpoint = endpoint or "/v1/chat/completions"
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    return base_url.rstrip("/") + endpoint


def detect_api_mode(endpoint: str) -> str:
    value = (endpoint or "").strip().lower()
    if "/responses" in value:
        return "responses"
    return "chat_completions"


def messages_to_responses_input(messages: List[Dict[str, Any]], *, system_role: str = "system") -> List[Dict[str, Any]]:
    input_items: List[Dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if not role:
            continue
        if role == "system" and system_role:
            role = system_role
        if "content" not in item:
            continue
        input_items.append({"type": "message", "role": role, "content": item.get("content")})
    return input_items


def get_api_key(config: Dict[str, Any], override_key: str) -> str:
    if override_key:
        return override_key
    env_key_name = (config.get("api_key_env") or "").strip()
    if not env_key_name:
        return ""
    return os.environ.get(env_key_name, "")


def build_headers(config: Dict[str, Any], api_key: str) -> Dict[str, str]:
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
    }
    extra_headers = config.get("headers") or {}
    if isinstance(extra_headers, dict):
        headers.update({str(k): str(v) for k, v in extra_headers.items()})

    if api_key:
        header_name = config.get("auth_header", "Authorization")
        prefix = config.get("auth_prefix", "Bearer")
        value = f"{prefix} {api_key}".strip() if prefix else api_key
        headers[str(header_name)] = value
    return headers


def parse_messages(
    use_raw: bool,
    raw_json: str,
    system_text: str,
    user_text: str,
) -> Optional[List[Dict[str, str]]]:
    if use_raw:
        if not raw_json.strip():
            st.error("Raw messages JSON is empty.")
            return None
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid raw messages JSON: {exc}")
            return None
        if not isinstance(data, list):
            st.error("Raw messages JSON must be a list.")
            return None
        return data

    messages: List[Dict[str, str]] = []
    if system_text.strip():
        messages.append({"role": "system", "content": system_text})
    if user_text.strip():
        messages.append({"role": "user", "content": user_text})
    else:
        st.error("User prompt is required when not using raw messages.")
        return None
    return messages


def extract_chat_completions_text(response_json: Dict[str, Any]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    contents: List[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and "content" in message:
            contents.append(str(message.get("content")))
    return "\n\n".join([c for c in contents if c])


def extract_responses_text(response_json: Dict[str, Any]) -> str:
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = response_json.get("output")
    if not isinstance(output, list) or not output:
        return ""

    parts: List[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        if item.get("role") != "assistant":
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "output_text" and "text" in block:
                parts.append(str(block.get("text")))
            elif block_type == "refusal" and "refusal" in block:
                parts.append(str(block.get("refusal")))
    return "\n\n".join([p for p in parts if p])


def parse_sse_events(raw_text: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    current_event: Optional[str] = None
    data_lines: List[str] = []

    def flush() -> None:
        nonlocal current_event, data_lines
        if current_event is None and not data_lines:
            return
        raw_data = "\n".join(data_lines).strip()
        parsed: Any = raw_data
        if raw_data and raw_data != "[DONE]":
            try:
                parsed = json.loads(raw_data)
            except json.JSONDecodeError:
                parsed = raw_data
        events.append({"event": current_event, "data": parsed, "raw_data": raw_data})
        current_event = None
        data_lines = []

    for line in (raw_text or "").splitlines():
        if not line.strip():
            flush()
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.lstrip()
        if field == "event":
            current_event = value
        elif field == "data":
            data_lines.append(value)

    flush()
    return events


def extract_assistant_text_from_sse(events: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []

    for event in events:
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        event_type = str(data.get("type") or "")
        if event_type == "response.output_text.delta":
            delta = data.get("delta")
            if isinstance(delta, str):
                chunks.append(delta)
        elif event_type == "response.output_text.done":
            text = data.get("text")
            if isinstance(text, str) and text.strip():
                return text
        elif event_type == "response.completed":
            response_obj = data.get("response")
            if isinstance(response_obj, dict):
                text = extract_responses_text(response_obj)
                if text:
                    return text

    text = "".join(chunks).strip()
    if text:
        return text

    for event in reversed(events):
        data = event.get("data")
        if isinstance(data, dict) and isinstance(data.get("response"), dict):
            text = extract_responses_text(data["response"])
            if text:
                return text

    return ""


def extract_charset(content_type: str) -> Optional[str]:
    match = re.search(r"charset=([^;\s]+)", content_type or "", flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().strip('"').strip("'")


def decode_http_body(body: bytes, content_type: str, default_encoding: str = "utf-8") -> tuple[str, str]:
    charset = extract_charset(content_type)
    if charset:
        try:
            return body.decode(charset, errors="replace"), charset
        except LookupError:
            pass

    try:
        return body.decode(default_encoding, errors="replace"), default_encoding
    except LookupError:
        return body.decode("utf-8", errors="replace"), "utf-8"


def extract_assistant_text(response_json: Dict[str, Any]) -> str:
    if not response_json:
        return ""
    return extract_responses_text(response_json) or extract_chat_completions_text(response_json)


def load_openai_standard_request() -> None:
    st.session_state["base_url_override"] = "https://api.openai.com"
    st.session_state["endpoint_override"] = "/v1/chat/completions"
    st.session_state["model_override"] = OPENAI_STANDARD_MODEL
    st.session_state["api_mode_choice"] = "Auto"
    st.session_state["use_raw_messages"] = True
    st.session_state["use_raw_request"] = False
    st.session_state["raw_messages_json"] = json.dumps(OPENAI_STANDARD_MESSAGES, indent=2)
    st.session_state["system_prompt"] = OPENAI_STANDARD_MESSAGES[0]["content"]
    st.session_state["user_prompt"] = OPENAI_STANDARD_MESSAGES[1]["content"]


def load_current_request_json() -> None:
    try:
        config_index = int(st.session_state.get("config_index", 0))
        config_file = Path(st.session_state.get("effective_config_file", "")).resolve()
        configs = load_configs(str(config_file))
        config = configs[config_index]

        default_params = config.get("default_params") or {}
        if not isinstance(default_params, dict):
            default_params = {}

        endpoint_override = str(st.session_state.get("endpoint_override", ""))
        model_override = str(st.session_state.get("model_override", ""))
        endpoint_default = str(config.get("endpoint", "/v1/chat/completions")).strip()
        effective_endpoint = endpoint_override.strip() or endpoint_default

        api_mode_auto = detect_api_mode(effective_endpoint)
        api_mode_choice = str(st.session_state.get("api_mode_choice", "Auto"))
        if api_mode_choice == "Responses":
            api_mode = "responses"
        elif api_mode_choice == "Chat Completions":
            api_mode = "chat_completions"
        else:
            api_mode = api_mode_auto

        use_raw_messages = bool(st.session_state.get("use_raw_messages", False))
        raw_messages_json = str(st.session_state.get("raw_messages_json", ""))
        system_prompt = str(st.session_state.get("system_prompt", ""))
        user_prompt = str(st.session_state.get("user_prompt", ""))

        extra_params = parse_extra_params(str(st.session_state.get(f"extra_params_json_{config_index}", "{}")))
        temperature = float(st.session_state.get(f"temperature_{config_index}", float(default_params.get("temperature", 0.7))))
        top_p = float(st.session_state.get(f"top_p_{config_index}", float(default_params.get("top_p", 1.0))))
        frequency_penalty = float(
            st.session_state.get(f"frequency_penalty_{config_index}", float(default_params.get("frequency_penalty", 0.0)))
        )
        presence_penalty = float(
            st.session_state.get(f"presence_penalty_{config_index}", float(default_params.get("presence_penalty", 0.0)))
        )
        max_tokens_default = int(default_params.get("max_tokens", default_params.get("max_output_tokens", 0)) or 0)
        max_tokens = int(st.session_state.get(f"max_tokens_{config_index}", max_tokens_default))

        params_from_file = dict(default_params)
        if api_mode == "responses":
            params_from_file.pop("max_tokens", None)
            params_from_file.pop("frequency_penalty", None)
            params_from_file.pop("presence_penalty", None)
        else:
            params_from_file.pop("max_output_tokens", None)

        payload: Dict[str, Any] = {}
        payload.update(params_from_file)
        payload["model"] = model_override

        if api_mode == "responses":
            payload["temperature"] = temperature
            payload["top_p"] = top_p
            if max_tokens > 0:
                payload["max_output_tokens"] = int(max_tokens)
            else:
                payload.pop("max_output_tokens", None)

            messages = parse_messages(use_raw_messages, raw_messages_json, system_prompt, user_prompt)
            if messages is None:
                return
            payload["input"] = messages_to_responses_input(messages, system_role=str(st.session_state.get("responses_system_role", "developer")))
        else:
            messages = parse_messages(use_raw_messages, raw_messages_json, system_prompt, user_prompt)
            if messages is None:
                return
            payload.update({
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            })
            if max_tokens > 0:
                payload["max_tokens"] = int(max_tokens)
            else:
                payload.pop("max_tokens", None)

        payload.update(extra_params)

        st.session_state["use_raw_request"] = True
        st.session_state["raw_request_json"] = json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as exc:
        _flash("error", f"Failed to build raw request JSON: {exc}")


st.set_page_config(page_title="LLM Debugger", layout="wide")
st.title("LLM API Debugger")
st.caption("Simple prompt debugger for OpenAI-compatible Chat Completions and Responses APIs.")

base_dir = Path(__file__).resolve().parent

st.sidebar.header("Config file")
_show_flash(st.sidebar)
config_path_text = st.sidebar.text_input(
    "Path",
    value=st.session_state.get("config_path_text", CONFIG_PATH),
    help="Relative paths are resolved against the folder containing app.py.",
    key="config_path_text",
)
config_path = Path(config_path_text)
config_file = config_path if config_path.is_absolute() else (base_dir / config_path).resolve()

configs = load_configs(str(config_file))
if not configs:
    st.error(f"No configs found in: {config_file}")
    st.stop()

config_names = [cfg.get("name", f"Config {i + 1}") for i, cfg in enumerate(configs)]

st.sidebar.header("Config")
config_index = st.sidebar.selectbox("Provider", list(range(len(configs))), format_func=lambda i: config_names[i])
st.session_state["config_index"] = int(config_index)
config = configs[config_index]
st.session_state["effective_config_file"] = str(config_file)

base_url = str(config.get("base_url", "")).strip()
endpoint = str(config.get("endpoint", "/v1/chat/completions")).strip()
model_default = str(config.get("model", "")).strip()
default_params = config.get("default_params") or {}
if not isinstance(default_params, dict):
    default_params = {}

timeout_default = int(config.get("timeout_seconds", DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT)
extra_params_default_obj = config.get("extra_params") or {}
if not isinstance(extra_params_default_obj, dict):
    extra_params_default_obj = {}
extra_params_default_text = json.dumps(extra_params_default_obj, ensure_ascii=False, indent=2)

active_identity = (str(config_file), int(config_index))
if st.session_state.get("_active_config_identity") != active_identity:
    st.session_state[f"temperature_{config_index}"] = float(default_params.get("temperature", 0.7))
    st.session_state[f"max_tokens_{config_index}"] = int(default_params.get("max_tokens", default_params.get("max_output_tokens", 0)) or 0)
    st.session_state[f"top_p_{config_index}"] = float(default_params.get("top_p", 1.0))
    st.session_state[f"frequency_penalty_{config_index}"] = float(default_params.get("frequency_penalty", 0.0))
    st.session_state[f"presence_penalty_{config_index}"] = float(default_params.get("presence_penalty", 0.0))
    st.session_state[f"timeout_seconds_{config_index}"] = int(timeout_default)
    st.session_state[f"extra_params_json_{config_index}"] = extra_params_default_text
    st.session_state["_active_config_identity"] = active_identity

base_url_override = st.sidebar.text_input(
    "Base URL override",
    value="",
    placeholder=base_url,
    key="base_url_override",
)
endpoint_override = st.sidebar.text_input(
    "Endpoint override",
    value="",
    placeholder=endpoint,
    key="endpoint_override",
)
effective_base_url = base_url_override.strip() or base_url
effective_endpoint = endpoint_override.strip() or endpoint

st.sidebar.caption(f"Request URL: {build_url(effective_base_url, effective_endpoint)}")

api_mode_auto = detect_api_mode(effective_endpoint)
api_mode_choice = st.sidebar.selectbox(
    "API mode",
    ["Auto", "Chat Completions", "Responses"],
    index=0,
    key="api_mode_choice",
    help="Auto infers from endpoint. Chat Completions uses `messages`; Responses uses role-based `input` messages.",
)
if api_mode_choice == "Responses":
    api_mode = "responses"
elif api_mode_choice == "Chat Completions":
    api_mode = "chat_completions"
else:
    api_mode = api_mode_auto
st.session_state["api_mode_effective"] = api_mode

if api_mode == "responses" and "/responses" not in effective_endpoint.lower():
    st.sidebar.warning("API mode is Responses but endpoint does not include `/responses`.")
elif api_mode == "chat_completions" and "/responses" in effective_endpoint.lower():
    st.sidebar.warning("API mode is Chat Completions but endpoint includes `/responses`.")
else:
    st.sidebar.caption(f"API mode: {api_mode}")

responses_system_role = str(st.session_state.get("responses_system_role", "developer"))
if api_mode == "responses":
    responses_system_role = st.sidebar.selectbox(
        "System role (Responses)",
        ["developer", "system"],
        index=0 if responses_system_role != "system" else 1,
        key="responses_system_role",
        help="Some OpenAI-compatible /v1/responses endpoints reject `system` messages. Use `developer` for compatibility.",
    )

api_key_override = st.sidebar.text_input("API key override", type="password")
model_override = st.sidebar.text_input("Model override", value=model_default, key="model_override")

with st.sidebar.expander("Import/Export configs", expanded=False):
    st.caption(f"Config file: {config_file}")

    export_source = st.radio(
        "Export source",
        ["Config file (no overrides)", "Current page (apply overrides + parameters to selected provider)"],
        index=0,
        key="export_source",
        help="Override/Parameters fields are not part of the config file unless you persist them.",
    )

    export_errors: List[str] = []
    if export_source == "Current page (apply overrides + parameters to selected provider)":
        export_configs = list(configs)
        updated = apply_provider_overrides(
            config,
            base_url_override=base_url_override,
            endpoint_override=endpoint_override,
            model_override=model_override,
        )
        try:
            current_extra_params = parse_extra_params(st.session_state.get(f"extra_params_json_{config_index}", "{}"))
        except ValueError as exc:
            export_errors.append(str(exc))
            current_extra_params = {}
        updated = apply_parameter_overrides(
            updated,
            api_mode=api_mode,
            temperature=float(st.session_state.get(f"temperature_{config_index}", 0.7)),
            max_tokens=int(st.session_state.get(f"max_tokens_{config_index}", 0) or 0),
            top_p=float(st.session_state.get(f"top_p_{config_index}", 1.0)),
            frequency_penalty=float(st.session_state.get(f"frequency_penalty_{config_index}", 0.0)),
            presence_penalty=float(st.session_state.get(f"presence_penalty_{config_index}", 0.0)),
            timeout_seconds=int(st.session_state.get(f"timeout_seconds_{config_index}", DEFAULT_TIMEOUT)),
            extra_params=current_extra_params,
        )
        export_configs[config_index] = updated
    else:
        export_configs = configs

    if export_errors:
        for msg in export_errors:
            st.error(msg)

    export_json = json.dumps(export_configs, ensure_ascii=False, indent=2) + "\n"
    st.download_button(
        "Download configs JSON",
        data=export_json,
        file_name=config_file.name or "configs.json",
        mime="application/json",
        use_container_width=True,
    )

    export_filename = st.text_input(
        "Export to file (relative)",
        value="configs.export.json",
        help="Saved under the folder containing app.py.",
        key="export_filename",
    )
    if st.button("Save export to disk", use_container_width=True, key="save_export"):
        try:
            target = resolve_target_path(base_dir, export_filename)
            write_json_file(target, export_configs)
            _flash("success", f"Exported configs to: {target}")
            st.rerun()
        except Exception as exc:
            st.error(f"Export failed: {exc}")

    if st.button(
        "Persist current page to config file",
        use_container_width=True,
        key="persist_overrides",
        help="Writes the selected provider's overrides + parameters into the active config file.",
    ):
        try:
            updated_configs = list(configs)
            updated = apply_provider_overrides(
                config,
                base_url_override=base_url_override,
                endpoint_override=endpoint_override,
                model_override=model_override,
            )
            updated = apply_parameter_overrides(
                updated,
                api_mode=api_mode,
                temperature=float(st.session_state.get(f"temperature_{config_index}", 0.7)),
                max_tokens=int(st.session_state.get(f"max_tokens_{config_index}", 0) or 0),
                top_p=float(st.session_state.get(f"top_p_{config_index}", 1.0)),
                frequency_penalty=float(st.session_state.get(f"frequency_penalty_{config_index}", 0.0)),
                presence_penalty=float(st.session_state.get(f"presence_penalty_{config_index}", 0.0)),
                timeout_seconds=int(st.session_state.get(f"timeout_seconds_{config_index}", DEFAULT_TIMEOUT)),
                extra_params=parse_extra_params(st.session_state.get(f"extra_params_json_{config_index}", "{}")),
            )
            updated_configs[config_index] = updated
            backup = backup_file(config_file)
            write_json_file(config_file, updated_configs)
            note = f" (backup: {backup})" if backup else ""
            _flash("success", f"Saved provider into: {config_file}{note}")
            st.rerun()
        except Exception as exc:
            st.error(f"Persist failed: {exc}")

    st.divider()

    uploaded = st.file_uploader("Import configs JSON", type=["json"], key="import_file")
    import_mode = st.radio(
        "Import mode",
        ["Overwrite current config file", "Save as new file"],
        index=0,
        key="import_mode",
    )
    import_filename = ""
    if import_mode == "Save as new file":
        import_filename = st.text_input(
            "Import target (relative)",
            value="configs.imported.json",
            key="import_filename",
        )

    if st.button(
        "Write imported config",
        disabled=uploaded is None,
        use_container_width=True,
        key="write_import",
    ):
        try:
            raw = uploaded.getvalue().decode("utf-8-sig")
            imported = validate_configs(json.loads(raw))

            if import_mode == "Overwrite current config file":
                target = config_file
                backup = backup_file(target)
                write_json_file(target, imported)
                note = f" (backup: {backup})" if backup else ""
                _flash("success", f"Imported configs to: {target}{note}")
                st.session_state.pop("_active_config_identity", None)
            else:
                target = resolve_target_path(base_dir, import_filename)
                write_json_file(target, imported)
                try:
                    st.session_state["config_path_text"] = str(target.relative_to(base_dir))
                except ValueError:
                    st.session_state["config_path_text"] = str(target)
                _flash("success", f"Imported configs to: {target}")
                st.session_state.pop("_active_config_identity", None)

            st.rerun()
        except Exception as exc:
            st.error(f"Import failed: {exc}")

st.sidebar.header("Parameters")
temperature = st.sidebar.number_input(
    "temperature",
    min_value=0.0,
    max_value=2.0,
    value=float(default_params.get("temperature", 0.7)),
    step=0.1,
    key=f"temperature_{config_index}",
)
max_tokens_label = "max_tokens" if api_mode == "chat_completions" else "max_output_tokens"
max_tokens_default = int(default_params.get("max_tokens", default_params.get("max_output_tokens", 0)) or 0)
max_tokens = st.sidebar.number_input(
    max_tokens_label,
    min_value=0,
    max_value=32000,
    value=int(max_tokens_default),
    step=1,
    key=f"max_tokens_{config_index}",
)
top_p = st.sidebar.number_input(
    "top_p",
    min_value=0.0,
    max_value=1.0,
    value=float(default_params.get("top_p", 1.0)),
    step=0.05,
    key=f"top_p_{config_index}",
)
frequency_penalty = st.sidebar.number_input(
    "frequency_penalty",
    min_value=-2.0,
    max_value=2.0,
    value=float(default_params.get("frequency_penalty", 0.0)),
    step=0.1,
    key=f"frequency_penalty_{config_index}",
    disabled=api_mode == "responses",
    help="Chat Completions only.",
)
presence_penalty = st.sidebar.number_input(
    "presence_penalty",
    min_value=-2.0,
    max_value=2.0,
    value=float(default_params.get("presence_penalty", 0.0)),
    step=0.1,
    key=f"presence_penalty_{config_index}",
    disabled=api_mode == "responses",
    help="Chat Completions only.",
)

timeout_seconds = st.sidebar.number_input(
    "timeout_seconds",
    min_value=1,
    max_value=300,
    value=int(timeout_default),
    step=1,
    key=f"timeout_seconds_{config_index}",
)

extra_params_json = st.sidebar.text_area(
    "extra params JSON",
    value=extra_params_default_text,
    height=140,
    key=f"extra_params_json_{config_index}",
)

st.header("Messages")
cols = st.columns(2)
cols[0].button("Load OpenAI standard request", on_click=load_openai_standard_request, use_container_width=True)
cols[1].button("Copy current request into raw JSON", on_click=load_current_request_json, use_container_width=True)
use_raw_request = st.checkbox("Use raw request JSON", key="use_raw_request")
use_raw_messages = st.checkbox("Use raw messages JSON", key="use_raw_messages")

system_prompt = st.text_area("System prompt", height=120, key="system_prompt")
user_prompt = st.text_area("User prompt", height=200, key="user_prompt")

raw_messages_json = ""
if use_raw_messages:
    raw_messages_json = st.text_area("Raw messages JSON", height=200, key="raw_messages_json")

raw_request_json = ""
if use_raw_request:
    st.caption("When enabled, the app sends exactly this JSON body; Parameters and Messages inputs are ignored.")
    raw_request_json = st.text_area("Raw request JSON", height=260, key="raw_request_json")

send = st.button("Send")

if send:
    if not effective_base_url:
        st.error("Base URL is missing in config.")
        st.stop()

    payload: Dict[str, Any] = {}
    if use_raw_request:
        if not raw_request_json.strip():
            st.error("Raw request JSON is empty.")
            st.stop()
        try:
            raw_payload = json.loads(raw_request_json)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid raw request JSON: {exc}")
            st.stop()
        if not isinstance(raw_payload, dict):
            st.error("Raw request JSON must be an object.")
            st.stop()
        payload = raw_payload
    else:
        if not model_override:
            st.error("Model is missing in config or override.")
            st.stop()

        try:
            extra_params = parse_extra_params(extra_params_json)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        params_from_file = dict(default_params)
        if api_mode == "responses":
            params_from_file.pop("max_tokens", None)
            params_from_file.pop("frequency_penalty", None)
            params_from_file.pop("presence_penalty", None)
        else:
            params_from_file.pop("max_output_tokens", None)

        payload.update(params_from_file)

        if api_mode == "responses":
            payload.update({
                "model": model_override,
                "temperature": temperature,
                "top_p": top_p,
            })

            if int(max_tokens) > 0:
                payload["max_output_tokens"] = int(max_tokens)
            else:
                payload.pop("max_output_tokens", None)

            messages = parse_messages(use_raw_messages, raw_messages_json, system_prompt, user_prompt)
            if messages is None:
                st.stop()
            payload["input"] = messages_to_responses_input(messages, system_role=str(st.session_state.get("responses_system_role", "developer")))
        else:
            messages = parse_messages(use_raw_messages, raw_messages_json, system_prompt, user_prompt)
            if messages is None:
                st.stop()

            payload.update({
                "model": model_override,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            })

            if int(max_tokens) > 0:
                payload["max_tokens"] = int(max_tokens)
            else:
                payload.pop("max_tokens", None)

        payload.update(extra_params)

    api_key = get_api_key(config, api_key_override)
    headers = build_headers(config, api_key)
    url = build_url(effective_base_url, effective_endpoint)

    st.subheader("Request JSON")
    st.json(payload)

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=float(timeout_seconds))
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
        st.stop()

    st.subheader("Response")
    st.write(f"HTTP {response.status_code}")

    response_json: Dict[str, Any] = {}
    content_type = str(response.headers.get("Content-Type", ""))
    if content_type:
        st.caption(f"Content-Type: {content_type}")

    raw_bytes = response.content or b""
    raw_text, decoded_encoding = decode_http_body(raw_bytes, content_type, default_encoding="utf-8")
    if decoded_encoding and "charset=" not in content_type.lower():
        st.caption(f"Decoded as: {decoded_encoding}")

    is_sse = "text/event-stream" in content_type.lower() or raw_bytes.lstrip().startswith(b"event:")

    if is_sse:
        events = parse_sse_events(raw_text)
        with st.expander("Stream events (parsed)", expanded=False):
            display_events = events if len(events) <= 50 else events[-50:]
            if len(events) > len(display_events):
                st.caption(f"Showing last {len(display_events)} of {len(events)} events.")
            st.json(display_events)

        with st.expander("Stream events (raw)", expanded=False):
            st.text_area("SSE", value=raw_text, height=260)

        final_response_obj: Optional[Dict[str, Any]] = None
        for event in reversed(events):
            data = event.get("data")
            if not isinstance(data, dict):
                continue
            if data.get("type") == "response.completed" and isinstance(data.get("response"), dict):
                final_response_obj = data["response"]
                break
            if isinstance(data.get("response"), dict):
                final_response_obj = data["response"]
                break

        if final_response_obj is not None:
            response_json = final_response_obj
            st.json(response_json)

        assistant_text = extract_assistant_text_from_sse(events) or extract_assistant_text(response_json)
    else:
        try:
            response_json = json.loads(raw_text)
            st.json(response_json)
        except ValueError:
            st.text(raw_text)
        assistant_text = extract_assistant_text(response_json)

    if assistant_text:
        st.subheader("Assistant Output")
        st.write(assistant_text)
