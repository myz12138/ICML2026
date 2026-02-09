
def is_var(x):
    return isinstance(x, str) and x.startswith("?")

def dedup_keep_order(xs):
    seen = set()
    out = []
    for x in xs:
        if not x:
            continue
        s = str(x).strip()
        if not s:
            continue
        low = s.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(s)
    return out

def get_var_candidate_names(bindings, var_name, max_k=5):
    info = (bindings or {}).get(var_name, {}) if isinstance(bindings, dict) else {}
    names = info.get("binding_names", []) or []
    names = [str(x).strip() for x in names if isinstance(x, str) and str(x).strip()]
    return dedup_keep_order(names)[: max(1, int(max_k))]

def get_known_entity_candidate_names(entity_map, surface):
    surface = (surface or "").strip()
    if not surface:
        return []
    names = [surface]
    if isinstance(entity_map, list):
        for rec in entity_map:
            if not isinstance(rec, dict):
                continue
            qe = (rec.get("query_entity") or "").strip()
            if qe == surface:
                for nm in (rec.get("kg_entity_names") or []):
                    if isinstance(nm, str) and nm.strip():
                        names.append(nm.strip())
                break
    return dedup_keep_order(names)

def get_candidate_names(bindings, entity_map, x):
    if is_var(x):
        return get_var_candidate_names(bindings, x)
    return get_known_entity_candidate_names(entity_map, x)

def build_query_texts(qt, bindings, entity_map):
    h = (qt.get("head") or "").strip()
    t = (qt.get("tail") or "").strip()

    rels = []
    rv = qt.get("relation_variants", [])
    if isinstance(rv, list):
        for x in rv:
            if isinstance(x, str) and x.strip():
                rels.append(x.strip())
    rels = dedup_keep_order(rels)

    head_opts = get_candidate_names(bindings, entity_map, h) if h else []
    tail_opts = get_candidate_names(bindings, entity_map, t) if t else []
    head_iter = head_opts if head_opts else [None]
    tail_iter = tail_opts if tail_opts else [None]

    qs = []

    def add_one(rp):
        rp = (rp or "").strip()
        if not rp:
            return
        for hh in head_iter:
            hh = (hh or "").strip() if isinstance(hh, str) else ""
            for tt in tail_iter:
                tt = (tt or "").strip() if isinstance(tt, str) else ""
                if hh and tt:
                    qs.append(f"{hh} {rp} {tt}")
                elif hh and not tt:
                    qs.append(f"{hh} {rp}")
                elif tt and not hh:
                    qs.append(f"{rp} {tt}")
                else:
                    qs.append(rp)

    for rp in rels:
        add_one(rp)

    qs = [" ".join(q.split()) for q in qs if q and str(q).strip()]
    return dedup_keep_order(qs)
