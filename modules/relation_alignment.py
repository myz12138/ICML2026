
from ..sim.embedding import text_sim

def clean_relation_variants(qt):
    rv = qt.get("relation_variants", [])
    if not isinstance(rv, list):
        rv = []
    out, seen = [], set()
    for x in rv:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if not s:
            continue
        low = s.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(s)
    return out

def relation_score(enc, qt, kg_relation):
    variants = clean_relation_variants(qt)
    if not variants:
        return 0.0
    kg_rel = str(kg_relation or "").strip()
    if not kg_rel:
        return 0.0
    return float(max(text_sim(enc, v, kg_rel) for v in variants))

def top_by_relation(enc, qt, triple_info, cand_ids, k=8):
    scored = []
    score_map = {}
    for tid in cand_ids:
        kg_rel = str(triple_info[tid]["relation"] or "")
        s = relation_score(enc, qt, kg_rel)
        score_map[int(tid)] = float(s)
        scored.append((s, tid))
    scored.sort(reverse=True)
    top_pairs = scored[: int(k)]
    top = [tid for s, tid in top_pairs]
    top_score_map = {int(tid): float(score_map.get(int(tid), 0.0)) for tid in top}
    return top, top_score_map
