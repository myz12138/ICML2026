
from collections import defaultdict

def is_var(x):
    return isinstance(x, str) and x.startswith("?")

def ent_name(e):
    if isinstance(e, dict):
        if e.get("name"):
            return e["name"]
        if e.get("surface"):
            return e["surface"]
    return str(e)

def parse_triple(tr, entities):
    if isinstance(tr, dict):
        h = tr.get("head", tr.get("h"))
        r = tr.get("relation", tr.get("r", ""))
        t = tr.get("tail", tr.get("t"))
    else:
        h, r, t = tr

    h_id = h if isinstance(h, int) and 0 <= h < len(entities) else None
    t_id = t if isinstance(t, int) and 0 <= t < len(entities) else None
    h_name = ent_name(entities[h_id]) if h_id is not None else str(h)
    t_name = ent_name(entities[t_id]) if t_id is not None else str(t)
    return h_id, str(r or ""), t_id, h_name, t_name

def build_kg_indices(kg):
    entities = kg["entities"]
    triples_raw = kg["triples"]

    triple_info = []
    out_index = defaultdict(list)
    in_index = defaultdict(list)

    for i, tr in enumerate(triples_raw):
        h_id, r, t_id, h_name, t_name = parse_triple(tr, entities)
        meta = {}
        if isinstance(tr, dict):
            for k in ("title", "doc_id", "source", "evidence", "sent_idx", "paragraph_idx"):
                if k in tr:
                    meta[k] = tr.get(k)
        triple_info.append(
            {
                "triple_id": int(i),
                "head_id": h_id,
                "tail_id": t_id,
                "relation": r,
                "head": h_name,
                "tail": t_name,
                "meta": meta,
            }
        )
        if isinstance(h_id, int):
            out_index[h_id].append(i)
        if isinstance(t_id, int):
            in_index[t_id].append(i)

    entity_names = [ent_name(e) for e in entities]
    return entities, entity_names, triple_info, out_index, in_index

def get_1hop_triples(eid, out_index, in_index):
    return list(set(out_index.get(eid, []) + in_index.get(eid, [])))
