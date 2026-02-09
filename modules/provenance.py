
import re

def _simple_norm(s):
    s = (s or "").lower()
    s = re.sub(r"[^0-9a-z\u4e00-\u9fff\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tri_in_text(text, tri, require_rel=False):
    sent = _simple_norm(text)
    h = _simple_norm(tri.get("head", ""))
    r = _simple_norm(tri.get("relation", ""))
    t = _simple_norm(tri.get("tail", ""))

    if not h or not t:
        return False
    ok = (h in sent) and (t in sent)
    if not ok:
        return False
    if require_rel:
        return bool(r) and (r in sent)
    return True
