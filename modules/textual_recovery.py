
from ..rerank.cross_encoder import score_pairs
from ..sim.embedding import text_sim
from .dataset_adapters import iter_context_units_2wiki_like, iter_paragraphs_musique
from .query_texts import build_query_texts

def build_rerank_query(question, qt, bindings, entity_map):
    q = (question or "").strip()
    qs = build_query_texts(qt, bindings, entity_map)
    lin_tau = qs[0] if qs else ""
    if q and lin_tau:
        return f"question: {q} constraint: {lin_tau}".strip()
    if q:
        return f"question: {q}".strip()
    return f"constraint: {lin_tau}".strip()

def retrieve_2wiki_like(enc, reranker, ex, question, qt, bindings, entity_map, topk,
                        doc_prefilter_k=80, unres_top_similar=20,
                        max_length=256, batch_size=32):
    topk = max(1, int(topk))
    q_texts = build_query_texts(qt, bindings, entity_map)
    rerank_query = build_rerank_query(question, qt, bindings, entity_map)

    all_cands = []
    for doc_idx, title, sents in iter_context_units_2wiki_like(ex):
        local = []
        for si, sent in enumerate(sents):
            filter_text = f"{title} {sent}".strip()
            if q_texts:
                coarse = max(text_sim(enc, q, filter_text) for q in q_texts)
            else:
                coarse = 0.0
            local.append((float(coarse), {
                "title": title,
                "paragraph_idx": int(doc_idx),
                "sent_idx": int(si),
                "context": filter_text,
                "evidence": sent,
                "similarity": float(coarse),
                "sentence_similarity": float(coarse),
            }))
        if local:
            local.sort(key=lambda x: x[0], reverse=True)
            local = local[: max(1, int(doc_prefilter_k))]
            all_cands.extend([x[1] for x in local])

    if not all_cands:
        return []

    all_cands.sort(key=lambda d: float(d["similarity"]), reverse=True)
    all_cands = all_cands[: max(1, int(unres_top_similar))]

    pairs = [(rerank_query, str(c.get("context") or "")) for c in all_cands]
    scores = score_pairs(reranker, pairs, max_length=max_length, batch_size=batch_size)
    for c, sc in zip(all_cands, scores):
        c["similarity"] = float(sc)
        c["sentence_similarity"] = float(sc)

    all_cands.sort(key=lambda d: float(d["similarity"]), reverse=True)
    return all_cands[:topk]

def retrieve_musique(enc, reranker, ex, question, qt, bindings, entity_map, topk,
                     unres_top_similar=20, max_length=256, batch_size=32):
    topk = max(1, int(topk))
    q_texts = build_query_texts(qt, bindings, entity_map)
    rerank_query = build_rerank_query(question, qt, bindings, entity_map)

    cands = []
    for para_idx, title, txt in iter_paragraphs_musique(ex):
        filter_text = f"{title} {txt}".strip()
        if q_texts:
            coarse = max(text_sim(enc, q, filter_text) for q in q_texts)
        else:
            coarse = 0.0
        cands.append({
            "title": title,
            "paragraph_idx": int(para_idx),
            "sent_idx": 0,
            "context": txt,
            "evidence": "",
            "similarity": float(coarse),
            "sentence_similarity": float(coarse),
        })

    if not cands:
        return []

    cands.sort(key=lambda d: float(d["similarity"]), reverse=True)
    cands = cands[: max(1, int(unres_top_similar))]

    pairs = [(rerank_query, str(c.get("context") or "")) for c in cands]
    scores = score_pairs(reranker, pairs, max_length=max_length, batch_size=batch_size)
    for c, sc in zip(cands, scores):
        c["similarity"] = float(sc)
        c["sentence_similarity"] = float(sc)

    cands.sort(key=lambda d: float(d["similarity"]), reverse=True)
    return cands[:topk]
