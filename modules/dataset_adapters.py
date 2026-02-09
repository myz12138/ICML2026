
def iter_context_units_2wiki_like(ex):
    ctx = ex.get("context") or ex.get("contexts") or []
    if not isinstance(ctx, list):
        return
    for doc_idx, item in enumerate(ctx):
        title = ""
        sents = []
        if isinstance(item, list) and len(item) >= 2:
            title = str(item[0] or "").strip()
            if isinstance(item[1], list):
                sents = [str(x or "").strip() for x in item[1] if str(x or "").strip()]
            else:
                blob = str(item[1] or "").strip()
                sents = [blob] if blob else []
        elif isinstance(item, dict):
            title = str(item.get("title") or "").strip()
            ss = item.get("sentences") or item.get("sents") or item.get("text") or []
            if isinstance(ss, list):
                sents = [str(x or "").strip() for x in ss if str(x or "").strip()]
            else:
                blob = str(ss or "").strip()
                sents = [blob] if blob else []
        else:
            continue
        if not title and not sents:
            continue
        yield doc_idx, title, sents

def iter_paragraphs_musique(ex):
    for p in ex.get("paragraphs", []) or []:
        if not isinstance(p, dict):
            continue
        title = str(p.get("title") or "").strip()
        para_idx = p.get("idx", None)
        txt = str(p.get("paragraph_text") or "").strip()
        if para_idx is None or not txt:
            continue
        yield int(para_idx), title, txt
