# LLM API Debugger

Simple Streamlit demo for debugging OpenAI-compatible chat APIs.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
streamlit run app.py
```

## Config

Edit `configs.json` to add providers. Each entry supports:

- `name`: label shown in UI
- `base_url`: API base URL
- `endpoint`: path for chat completions (default `/v1/chat/completions`)
- `model`: default model
- `api_key_env`: environment variable for the API key
- `auth_header`: header name for auth (default `Authorization`)
- `auth_prefix`: prefix for auth header (default `Bearer`)
- `headers`: extra headers object
- `default_params`: default payload params merged into request

You can override the config file path with `LLM_DEBUG_CONFIG`.
