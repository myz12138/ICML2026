
from ..rerank.cross_encoder import score_pairs

def lin_query_triple(qt, max_vars=3):
    h = str(qt.get("head", "") or "").strip()
    t = str(qt.get("tail", "") or "").strip()
    rv = qt.get("relation_variants", [])
    if not isinstance(rv, list):
        rv = []
    rv = [str(x).strip() for x in rv if isinstance(x, str) and str(x).strip()]
    rv = rv[: max(1, int(max_vars))]
    rtxt = " | ".join(rv)
    if rtxt:
        return " ".join([x for x in [h, f"({rtxt})", t] if x]).strip()
    return " ".join([x for x in [h, t] if x]).strip()

def lin_kg_triple(tr):
    h = str(tr.get("head", "") or "").strip()
    r = str(tr.get("relation", "") or "").strip()
    t = str(tr.get("tail", "") or "").strip()
    return " ".join([x for x in [h, r, t] if x]).strip()

def rerank_select_topn(reranker, question, qt, candidates, top_n=3,
                       include_question=True, max_vars_in_lin=3,
                       max_length=128, batch_size=32):
    if not candidates:
        return [], [], {}

    q_triple_str = lin_query_triple(qt, max_vars=max_vars_in_lin)
    q_prefix = (str(question or "").strip() + " ") if include_question else ""
    query_text = ("question: " + q_prefix + " UnknownTriple: " + q_triple_str).strip()

    pairs = []
    for c in candidates:
        pairs.append((query_text, lin_kg_triple(c)))

    scores = score_pairs(reranker, pairs, max_length=max_length, batch_size=batch_size)

    scored = [(float(scores[i]), i) for i in range(len(scores))]
    scored.sort(reverse=True)
    keep = scored[: max(1, int(top_n))]
    selected_indices = [idx for s, idx in keep]
    selected_triple_ids = [candidates[idx]["triple_id"] for idx in selected_indices]
    score_map = {int(candidates[i]["triple_id"]): float(scores[i]) for i in range(len(candidates))}
    return selected_triple_ids, selected_indices, score_map
