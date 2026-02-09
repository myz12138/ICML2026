
from ..sim.string_match import str_sim, has_token_overlap

def map_query_entities(query_triples, entity_names, top_k_entity=3):
    ents = set()
    for qt in query_triples:
        h = qt.get("head")
        t = qt.get("tail")
        if h and not (isinstance(h, str) and h.startswith("?")):
            ents.add(h)
        if t and not (isinstance(t, str) and t.startswith("?")):
            ents.add(t)

    mapping = {}
    for qe in ents:
        scored = []
        for idx, name in enumerate(entity_names):
            if not has_token_overlap(qe, name):
                continue
            s = str_sim(qe, name)
            scored.append((s, idx))
        if not scored:
            continue
        scored.sort(reverse=True)
        mapping[qe] = [idx for s, idx in scored[: int(top_k_entity)]]
    return list(ents), mapping

def map_surface_to_entity_ids(surface, entity_names, top_k_entity=3):
    surface = str(surface)
    scored = []
    for idx, name in enumerate(entity_names):
        if not has_token_overlap(surface, name):
            continue
        s = str_sim(surface, name)
        scored.append((s, idx))
    if not scored:
        return []
    scored.sort(reverse=True)
    return [idx for s, idx in scored[: int(top_k_entity)]]
