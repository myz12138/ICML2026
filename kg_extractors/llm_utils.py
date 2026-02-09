import os, json, requests

class LLMClient:
    def __init__(self):
        self.endpoint = os.getenv("LLM_ENDPOINT", "")
        self.api_key  = os.getenv("LLM_API_KEY",'')
        self.model    = os.getenv("LLM_MODEL", "gpt-4o-mini")
        if not (self.endpoint and self.api_key):
            raise RuntimeError("LLM endpoint/key not set.")

        self.endpoint = self.endpoint.rstrip('/') + '/'

    def chat(self, messages, temperature=0.2, max_tokens=256):
        url = self.endpoint
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "temperature": float(temperature), "max_tokens": int(max_tokens)}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


import os, re, json

class LLMClient:
    def __init__(self, api_key: str = None, base_url: str = None,
                 model: str = None, temperature: float = None, max_tokens: int = None):
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError(f"需要安装 openai：pip install 'openai>=1.0.0' ；错误：{e}")

        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("LLM_ENDPOINT") or os.environ.get("OPENAI_BASE_URL","")
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.default_temperature = float( (temperature if temperature is not None else os.environ.get("LLM_TEMPERATURE", 0.2)) )
        self.default_max_tokens  = int( (max_tokens if max_tokens is not None else os.environ.get("LLM_MAX_TOKENS", 256)) )


        b = self.base_url.rstrip("/")
        if not b.endswith("/v1"):
            if not b.endswith("/v1/chat/completions"):
                b = b + "/v1"
        self.base_url = b

        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def ask_json(self, system_prompt: str, user_prompt: str):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.default_temperature,
                messages=[
                    {"role":"system","content":system_prompt},
                    {"role":"user","content":user_prompt},
                ],
                max_tokens=self.default_max_tokens,
            )
            content = resp.choices[0].message.content.strip()
            m = re.search(r"\{[\s\S]*\}$|^\[[\s\S]*\]$", content)
            if m:
                content = m.group(0)
            return json.loads(content)
        except Exception:
            return {}

    def chat(self, messages, temperature=None, max_tokens=None):
        t = self.default_temperature if temperature is None else float(temperature)
        mt = self.default_max_tokens  if max_tokens  is None else int(max_tokens)
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=t,
            messages=messages,
            max_tokens=mt,
        )
        content = ""
        try:
            content = resp.choices[0].message.content
        except Exception:
            content = getattr(resp.choices[0], "text", "")
        return (content or "").strip()