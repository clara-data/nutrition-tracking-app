import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3"


def _normalize_messages(payload):
    if isinstance(payload, str):
        return [{"role": "user", "content": payload}]
    if isinstance(payload, list):
        return payload
    raise ValueError("LLM input must be either a string or a list of messages.")


def _call_ollama(payload_input):
    messages = _normalize_messages(payload_input)

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)

    if not response.ok:
        raise RuntimeError(
            f"Ollama request failed ({response.status_code}): {response.text}"
        )

    data = response.json()
    return data.get("message", {}).get("content", "")


def call_meal_parser_llm(messages_or_prompt):
    return _call_ollama(messages_or_prompt)


def call_dietary_advice_llm(messages_or_prompt):
    return _call_ollama(messages_or_prompt)