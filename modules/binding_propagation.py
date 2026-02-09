
def is_var(x):
    return isinstance(x, str) and x.startswith("?")

def induced_entities_for_single_var(qt, triple_info, selected_tids, anchor_nodes):
    h = qt.get("head")
    t = qt.get("tail")
    is_hv = is_var(h)
    is_tv = is_var(t)
    if not (is_hv ^ is_tv):
        return None, set()

    var_name = h if is_hv else t
    anchor_set = set(anchor_nodes or [])

    out = set()
    for tid in selected_tids:
        info = triple_info[int(tid)]
        hi = info.get("head_id")
        ti = info.get("tail_id")
        if isinstance(hi, int) and hi in anchor_set and isinstance(ti, int) and ti not in anchor_set:
            out.add(int(ti))
        if isinstance(ti, int) and ti in anchor_set and isinstance(hi, int) and hi not in anchor_set:
            out.add(int(hi))
    return var_name, out

def consolidate_bindings(var_to_sets):
    out = {}
    for v, sets in (var_to_sets or {}).items():
        sets = [set(s) for s in (sets or []) if s]
        union_set = set()
        for s in sets:
            union_set |= set(s)

        inter_set = None
        if len(sets) > 1:
            inter_set = set.intersection(*sets) if sets else set()

        if inter_set is not None and len(inter_set) > 0:
            binding = set(inter_set)
            method = "intersection"
        else:
            binding = set(union_set)
            method = "union"
        out[v] = {"union": union_set, "intersection": inter_set, "binding": binding, "method": method}
    return out

def filter_triples_by_binding(qt, triple_info, selected_tids, anchor_nodes, binding_set):
    var_name, induced = induced_entities_for_single_var(qt, triple_info, selected_tids, anchor_nodes)
    if not var_name:
        return selected_tids
    bset = set(binding_set or [])
    kept = []
    anchor_set = set(anchor_nodes or [])
    for tid in selected_tids:
        info = triple_info[int(tid)]
        hi = info.get("head_id")
        ti = info.get("tail_id")
        cand = None
        if isinstance(hi, int) and hi in anchor_set and isinstance(ti, int) and ti not in anchor_set:
            cand = int(ti)
        if isinstance(ti, int) and ti in anchor_set and isinstance(hi, int) and hi not in anchor_set:
            cand = int(hi)
        if cand is None:
            continue
        if cand in bset:
            kept.append(int(tid))
    return kept
