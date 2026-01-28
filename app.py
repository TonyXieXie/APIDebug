import json
import os
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


def load_configs(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        st.error(f"Config file not found: {path}")
        st.stop()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON in config file: {exc}")
        st.stop()
    if not isinstance(data, list):
        st.error("Config file must contain a JSON list of configs.")
        st.stop()
    return data


def build_url(base_url: str, endpoint: str) -> str:
    endpoint = endpoint or "/v1/chat/completions"
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    return base_url.rstrip("/") + endpoint


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


def extract_assistant_text(response_json: Dict[str, Any]) -> str:
    if not response_json:
        return ""
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


def load_openai_standard_request() -> None:
    st.session_state["base_url_override"] = "https://api.openai.com"
    st.session_state["endpoint_override"] = "/v1/chat/completions"
    st.session_state["model_override"] = OPENAI_STANDARD_MODEL
    st.session_state["use_raw_messages"] = True
    st.session_state["raw_messages_json"] = json.dumps(OPENAI_STANDARD_MESSAGES, indent=2)
    st.session_state["system_prompt"] = OPENAI_STANDARD_MESSAGES[0]["content"]
    st.session_state["user_prompt"] = OPENAI_STANDARD_MESSAGES[1]["content"]


st.set_page_config(page_title="LLM Debugger", layout="wide")
st.title("LLM API Debugger")
st.caption("Simple prompt debugger for OpenAI-compatible chat APIs.")

configs = load_configs(CONFIG_PATH)
config_names = [cfg.get("name", f"Config {i + 1}") for i, cfg in enumerate(configs)]

st.sidebar.header("Config")
config_index = st.sidebar.selectbox("Provider", list(range(len(configs))), format_func=lambda i: config_names[i])
config = configs[config_index]

base_url = str(config.get("base_url", "")).strip()
endpoint = str(config.get("endpoint", "/v1/chat/completions")).strip()
model_default = str(config.get("model", "")).strip()
default_params = config.get("default_params") or {}
if not isinstance(default_params, dict):
    default_params = {}

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

api_key_override = st.sidebar.text_input("API key override", type="password")
model_override = st.sidebar.text_input("Model override", value=model_default, key="model_override")

st.sidebar.header("Parameters")
temperature = st.sidebar.number_input(
    "temperature",
    min_value=0.0,
    max_value=2.0,
    value=float(default_params.get("temperature", 0.7)),
    step=0.1,
    key=f"temperature_{config_index}",
)
max_tokens = st.sidebar.number_input(
    "max_tokens",
    min_value=0,
    max_value=32000,
    value=int(default_params.get("max_tokens", 0)),
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
)
presence_penalty = st.sidebar.number_input(
    "presence_penalty",
    min_value=-2.0,
    max_value=2.0,
    value=float(default_params.get("presence_penalty", 0.0)),
    step=0.1,
    key=f"presence_penalty_{config_index}",
)

timeout_seconds = st.sidebar.number_input("timeout_seconds", min_value=1, max_value=300, value=DEFAULT_TIMEOUT, step=1)

extra_params_json = st.sidebar.text_area("extra params JSON", value="{}", height=140, key="extra_params_json")

st.header("Messages")
st.button("Load OpenAI standard request", on_click=load_openai_standard_request)
use_raw_messages = st.checkbox("Use raw messages JSON", key="use_raw_messages")

system_prompt = st.text_area("System prompt", height=120, key="system_prompt")
user_prompt = st.text_area("User prompt", height=200, key="user_prompt")

raw_messages_json = ""
if use_raw_messages:
    raw_messages_json = st.text_area("Raw messages JSON", height=200, key="raw_messages_json")

send = st.button("Send")

if send:
    if not effective_base_url:
        st.error("Base URL is missing in config.")
        st.stop()
    if not model_override:
        st.error("Model is missing in config or override.")
        st.stop()

    messages = parse_messages(use_raw_messages, raw_messages_json, system_prompt, user_prompt)
    if messages is None:
        st.stop()

    try:
        extra_params = json.loads(extra_params_json) if extra_params_json.strip() else {}
    except json.JSONDecodeError as exc:
        st.error(f"Invalid extra params JSON: {exc}")
        st.stop()

    if not isinstance(extra_params, dict):
        st.error("Extra params JSON must be an object.")
        st.stop()

    payload: Dict[str, Any] = {}
    payload.update(default_params)

    payload.update({
        "model": model_override,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    })

    if max_tokens:
        payload["max_tokens"] = int(max_tokens)

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
    try:
        response_json = response.json()
        st.json(response_json)
    except ValueError:
        st.text(response.text)

    assistant_text = extract_assistant_text(response_json)
    if assistant_text:
        st.subheader("Assistant Output")
        st.write(assistant_text)
