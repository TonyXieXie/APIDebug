# LLM API Debugger

Simple Streamlit demo for debugging OpenAI-compatible Chat Completions and Responses APIs.

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
- `endpoint`: request path (e.g. `/v1/chat/completions` or `/v1/responses`)
- `model`: default model
- `api_key_env`: environment variable for the API key
- `auth_header`: header name for auth (default `Authorization`)
- `auth_prefix`: prefix for auth header (default `Bearer`)
- `headers`: extra headers object
- `default_params`: default payload params merged into request
- `timeout_seconds`: default request timeout for this provider
- `extra_params`: default extra params object merged into request (loaded into the “extra params JSON” box)

You can override the config file path with `LLM_DEBUG_CONFIG`.

## API Mode

The sidebar **API mode** controls how the request payload is built (and how the output text is extracted):

- **Chat Completions**: sends `messages` to `/v1/chat/completions` (assistant text from `choices[].message.content`)
- **Responses**: sends `input`/`instructions` to `/v1/responses` (assistant text from `output[].content[].text`)

**Auto** infers the mode from the endpoint path (if it contains `/responses`, it assumes Responses).

## Import/Export

In the sidebar, you can change the active config file via **Config file → Path**.
Relative paths are resolved against the folder containing `app.py`.

Use **Import/Export configs** to:

- Download the current configs JSON (either from the config file, or with the current page overrides + parameters applied)
- Save an export copy to disk (relative to the folder containing `app.py`)
- Import a JSON file and overwrite `configs.json` (with an automatic `.bak.*` backup) or save as a new file

## Raw Request JSON

In **Messages**, enable **Use raw request JSON** to send a full request body as-is.
Use **Copy current request into raw JSON** to generate a starting point from the current UI settings.
