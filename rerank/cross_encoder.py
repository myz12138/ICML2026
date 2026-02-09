import torch

# Try transformers CrossEncoder-style scoring; otherwise fall back to a no-op scorer.
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _HAS_TRANSFORMERS = True
except Exception:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    _HAS_TRANSFORMERS = False

def build_reranker(model_name, fp16=True):
    if not _HAS_TRANSFORMERS:
        return {"kind": "none", "model_name": model_name, "device": "cpu"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    mdl.eval()
    mdl.to(device)
    if device.startswith("cuda") and fp16:
        mdl.half()
    return {"kind": "hf", "tokenizer": tok, "model": mdl, "device": device}

def score_pairs(reranker, pairs, max_length=128, batch_size=32):
    if not pairs:
        return []
    if not reranker or reranker.get("kind") != "hf":
        # No reranker available -> return zeros (pipeline still runs; quality will drop)
        return [0.0 for _ in pairs]

    tok = reranker["tokenizer"]
    mdl = reranker["model"]
    device = reranker["device"]
    scores = []
    bs = max(1, int(batch_size))
    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            batch = pairs[i : i + bs]
            qs = [p[0] for p in batch]
            ps = [p[1] for p in batch]
            encd = tok(qs, ps, padding=True, truncation=True, max_length=int(max_length), return_tensors="pt")
            encd = {k: v.to(device) for k, v in encd.items()}
            out = mdl(**encd)
            logits = out.logits
            if logits.ndim == 2 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            logits = logits.float().detach().cpu().tolist()
            if isinstance(logits, float):
                logits = [logits]
            scores.extend([float(x) for x in logits])
    return scores
