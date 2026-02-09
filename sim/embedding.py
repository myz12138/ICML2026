import torch
import torch.nn.functional as F

# Prefer your existing EmbeddingClient if present; otherwise fall back to a local encoder.
try:
    from utils.remote_emb import EmbeddingClient  # your repo
except Exception:
    EmbeddingClient = None

_TEXT_SIM_CACHE = {}
_TEXT_SIM_MAX = 50000

class _LocalEmbeddingClient:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self._kind = None
        self._model = None
        self._tokenizer = None

        # sentence-transformers (fast path)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, device=device)
            self._kind = "st"
            return
        except Exception:
            pass

        # transformers fallback
        try:
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name).to(device)
            self._model.eval()
            self._kind = "hf"
        except Exception as e:
            raise RuntimeError(
                "No embedding backend available. Install sentence-transformers or transformers, "
                "or provide utils.remote_emb.EmbeddingClient."
            ) from e

    def encode_one(self, text):
        text = str(text or "")
        if self._kind == "st":
            vec = self._model.encode([text], normalize_embeddings=True, convert_to_tensor=True)[0]
            return vec.detach().cpu()
        # hf mean pooling
        tok = self._tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        tok = {k: v.to(self.device) for k, v in tok.items()}
        with torch.no_grad():
            out = self._model(**tok)
            last = out.last_hidden_state  # [B,T,H]
            mask = tok.get("attention_mask", None)
            if mask is None:
                pooled = last.mean(dim=1)
            else:
                mask = mask.unsqueeze(-1).float()
                pooled = (last * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1e-6))
        pooled = pooled[0].detach().cpu()
        pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled

def build_embedder(emb_model, emb_device):
    if EmbeddingClient is not None:
        return EmbeddingClient(emb_model, emb_device)
    return _LocalEmbeddingClient(emb_model, emb_device)

def text_sim(enc, a, b):
    if not a or not b:
        return 0.0
    a = str(a)
    b = str(b)

    def get_vec(x):
        v = _TEXT_SIM_CACHE.get(x)
        if v is not None:
            return v
        vec = enc.encode_one(x)
        if not isinstance(vec, torch.Tensor):
            vec = torch.tensor(vec, dtype=torch.float32)
        vec = F.normalize(vec.unsqueeze(0), p=2, dim=-1).squeeze(0)

        if len(_TEXT_SIM_CACHE) >= _TEXT_SIM_MAX:
            try:
                _TEXT_SIM_CACHE.pop(next(iter(_TEXT_SIM_CACHE)))
            except Exception:
                _TEXT_SIM_CACHE.clear()
        _TEXT_SIM_CACHE[x] = vec
        return vec

    v1 = get_vec(a)
    v2 = get_vec(b)
    return float(torch.dot(v1, v2).item())
