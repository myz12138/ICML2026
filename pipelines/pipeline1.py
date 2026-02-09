
from collections import defaultdict

from ..kg.index import is_var, ent_name, get_1hop_triples
from .anchor_matching import map_query_entities, map_surface_to_entity_ids
from .relation_alignment import top_by_relation
from .contextual_reranking import rerank_select_topn
from .sufficiency_check import neff_from_logits, is_resolved
from .binding_propagation import induced_entities_for_single_var, consolidate_bindings, filter_triples_by_binding

def run_constraint_retrieval_for_triple(
    enc,
    reranker,
    entities,
    entity_names,
    triple_info,
    out_index,
    in_index,
    question,
    qt_effective,
    anchor_nodes,
    args,
    stage_tag,
    triple_type_initial,
):
    cand_ids_all = set()
    for nid in anchor_nodes or []:
        cand_ids_all.update(get_1hop_triples(int(nid), out_index, in_index))
    cand_ids_all = sorted(cand_ids_all)

    cand_ids_top, rel_score_map = ([], {})
    if cand_ids_all:
        cand_ids_top, rel_score_map = top_by_relation(enc, qt_effective, triple_info, cand_ids_all, k=args.relation_k)

    selected_tids, selected_indices, rerank_score_map = ([], [], {})
    neff_val = float("inf")
    status = "unresolved"

    if cand_ids_top:
        candidates = [
            {"triple_id": int(tid), "head": triple_info[tid]["head"], "relation": triple_info[tid]["relation"], "tail": triple_info[tid]["tail"]}
            for tid in cand_ids_top
        ]

        selected_tids, selected_indices, rerank_score_map = rerank_select_topn(
            reranker,
            question,
            qt_effective,
            candidates,
            top_n=args.top_n,
            include_question=bool(args.include_question_in_rerank),
            max_length=args.rerank_max_length,
            batch_size=args.rerank_batch,
        )

        logits = [float(rerank_score_map.get(int(tid), 0.0)) for tid in selected_tids]
        neff_val = neff_from_logits(logits)
        status = "resolved" if is_resolved(neff_val, args.neff_threshold) else "unresolved"

    debug = {
        "stage": stage_tag,
        "triple_type_initial": triple_type_initial,
        "anchors": [{"entity_id": int(nid), "entity_name": entity_names[int(nid)]} for nid in (anchor_nodes or [])],
        "candidate_ids_topk": [int(tid) for tid in cand_ids_top],
        "selected_triple_ids": [int(tid) for tid in selected_tids],
        "selected_indices_in_topk": selected_indices,
        "neff": float(neff_val),
        "status": status,
    }

    rel_scores_selected = {int(tid): float(rel_score_map.get(int(tid), 0.0)) for tid in selected_tids}
    rerank_scores_selected = {int(tid): float(rerank_score_map.get(int(tid), 0.0)) for tid in selected_tids}

    return {
        "selected_triple_ids": [int(x) for x in selected_tids],
        "rel_scores_selected": rel_scores_selected,
        "rerank_scores_selected": rerank_scores_selected,
        "neff": float(neff_val),
        "status": status,
        "debug": debug,
    }

def process_item_stage1(item, kg, entities, entity_names, triple_info, out_index, in_index, enc, reranker, args):
    question = item.get("question", "")
    qtriples = item.get("query_plan") or []

    known_entities, emap = map_query_entities(qtriples, entity_names, top_k_entity=args.top_k_entity)

    triple_meta = []
    for qi, qt in enumerate(qtriples):
        h = qt.get("head")
        t = qt.get("tail")
        is_hv = is_var(h)
        is_tv = is_var(t)

        anchor_nodes = []
        if (not is_hv) and h in emap:
            anchor_nodes.extend(emap[h])
        if (not is_tv) and t in emap:
            for eid in emap[t]:
                if eid not in anchor_nodes:
                    anchor_nodes.append(eid)

        has_anchor = len(anchor_nodes) > 0
        if is_hv and is_tv:
            ttype = "double_var"
        elif (is_hv ^ is_tv) and has_anchor:
            ttype = "single_var_with_anchor"
        elif (not is_hv and not is_tv) and has_anchor:
            ttype = "no_var_with_anchor"
        else:
            ttype = "no_anchor"

        triple_meta.append({"index": qi, "anchor_nodes_initial": anchor_nodes, "triple_type_initial": ttype})

    per_triple = [None] * len(qtriples)
    var_to_induced_sets = defaultdict(list)
    support = defaultdict(list)

    # First pass
    for meta in triple_meta:
        qi = meta["index"]
        qt = qtriples[qi]
        anchor_nodes = meta["anchor_nodes_initial"]
        ttype = meta["triple_type_initial"]

        record = {
            "query_triple_index": int(qi),
            "query_triple": qt,
            "triple_type_initial": ttype,
            "stage_records": [],
            "status": "unresolved",
        }

        if (not anchor_nodes) or (ttype == "double_var"):
            per_triple[qi] = record
            continue

        out = run_constraint_retrieval_for_triple(
            enc, reranker, entities, entity_names, triple_info, out_index, in_index,
            question, qt, anchor_nodes, args, "first_pass", ttype
        )
        record["stage_records"].append(out["debug"])
        record["status"] = out["status"]
        record["selected_triple_ids"] = out["selected_triple_ids"]
        record["neff"] = out["neff"]

        if out["status"] == "resolved" and out["selected_triple_ids"]:
            support[qi].extend(out["selected_triple_ids"])
            v, induced = induced_entities_for_single_var(qt, triple_info, out["selected_triple_ids"], anchor_nodes)
            record["induced_entities"] = sorted([int(x) for x in induced]) if induced else []
            if v and induced:
                var_to_induced_sets[v].append(set(induced))
        else:
            record["induced_entities"] = []

        per_triple[qi] = record

    # Consolidate bindings
    bindings_raw = consolidate_bindings(var_to_induced_sets)
    bindings = {}
    for v, info in bindings_raw.items():
        bset = info.get("binding", set()) or set()
        bindings[v] = {
            "binding_ids": sorted([int(x) for x in bset]),
            "binding_names": [entity_names[int(x)] for x in sorted([int(x) for x in bset]) if 0 <= int(x) < len(entity_names)],
            "method": info.get("method"),
            "union_ids": sorted([int(x) for x in (info.get("union") or set())]),
            "intersection_ids": sorted([int(x) for x in (info.get("intersection") or set())]) if info.get("intersection") is not None else None,
        }

    # Filter resolved single-var selections by binding consistency
    for meta in triple_meta:
        qi = meta["index"]
        ttype = meta["triple_type_initial"]
        if ttype != "single_var_with_anchor":
            continue
        rec = per_triple[qi]
        if rec.get("status") != "resolved":
            continue
        qt = rec.get("query_triple")
        anchor_nodes = meta["anchor_nodes_initial"]
        h = qt.get("head"); t = qt.get("tail")
        v = h if is_var(h) else (t if is_var(t) else None)
        if not v or v not in bindings:
            continue
        kept = filter_triples_by_binding(qt, triple_info, rec.get("selected_triple_ids") or [], anchor_nodes, bindings[v]["binding_ids"])
        rec["selected_triple_ids_filtered"] = [int(x) for x in kept]
        support[qi] = [int(x) for x in kept]

    # Alias-expanded anchors for propagation
    var_candidate_anchors = defaultdict(dict)
    for v, info in bindings.items():
        for eid in info.get("binding_ids", []) or []:
            anchor_ids = set([int(eid)])
            surf = ent_name(entities[int(eid)]) if 0 <= int(eid) < len(entities) else str(eid)
            for mid in map_surface_to_entity_ids(surf, entity_names, top_k_entity=args.top_k_entity):
                anchor_ids.add(int(mid))
            var_candidate_anchors[v][int(eid)] = sorted(anchor_ids)

    # Second pass: double-var propagation
    for meta in triple_meta:
        qi = meta["index"]
        if meta["triple_type_initial"] != "double_var":
            continue

        qt = qtriples[qi]
        h = qt.get("head"); t = qt.get("tail")
        h_var = h if is_var(h) else None
        t_var = t if is_var(t) else None
        h_bind = bindings.get(h_var, {}).get("binding_ids", []) if h_var else []
        t_bind = bindings.get(t_var, {}).get("binding_ids", []) if t_var else []

        record = per_triple[qi] or {
            "query_triple_index": int(qi),
            "query_triple": qt,
            "triple_type_initial": "double_var",
            "stage_records": [],
            "status": "unresolved",
        }

        if not h_bind and not t_bind:
            per_triple[qi] = record
            continue

        selected_all = []

        def _instantiate_qt(head_val=None, tail_val=None):
            new_qt = {"head": qt.get("head"), "tail": qt.get("tail"), "relation_variants": qt.get("relation_variants", [])}
            if head_val is not None:
                new_qt["head"] = head_val
            if tail_val is not None:
                new_qt["tail"] = tail_val
            return new_qt

        if h_bind and not t_bind:
            for hb in h_bind[: max(1, int(args.top_n))]:
                anchor_nodes = var_candidate_anchors.get(h_var, {}).get(int(hb), [])
                if not anchor_nodes:
                    continue
                new_qt = _instantiate_qt(head_val=entity_names[int(hb)])
                out = run_constraint_retrieval_for_triple(
                    enc, reranker, entities, entity_names, triple_info, out_index, in_index,
                    question, new_qt, anchor_nodes, args, "second_pass", "double_var"
                )
                record["stage_records"].append(out["debug"])
                if out["selected_triple_ids"]:
                    selected_all.extend(out["selected_triple_ids"])
                if out["status"] == "resolved":
                    record["status"] = "resolved"
                    v2, induced = induced_entities_for_single_var(new_qt, triple_info, out["selected_triple_ids"], anchor_nodes)
                    if v2 and induced:
                        var_to_induced_sets[v2].append(set(induced))

        elif t_bind and not h_bind:
            for tb in t_bind[: max(1, int(args.top_n))]:
                anchor_nodes = var_candidate_anchors.get(t_var, {}).get(int(tb), [])
                if not anchor_nodes:
                    continue
                new_qt = _instantiate_qt(tail_val=entity_names[int(tb)])
                out = run_constraint_retrieval_for_triple(
                    enc, reranker, entities, entity_names, triple_info, out_index, in_index,
                    question, new_qt, anchor_nodes, args, "second_pass", "double_var"
                )
                record["stage_records"].append(out["debug"])
                if out["selected_triple_ids"]:
                    selected_all.extend(out["selected_triple_ids"])
                if out["status"] == "resolved":
                    record["status"] = "resolved"
                    v2, induced = induced_entities_for_single_var(new_qt, triple_info, out["selected_triple_ids"], anchor_nodes)
                    if v2 and induced:
                        var_to_induced_sets[v2].append(set(induced))
        else:
            # both sides bound: optional evidence lookup
            for hb in h_bind[: max(1, int(args.top_n))]:
                for tb in t_bind[: max(1, int(args.top_n))]:
                    anchor_nodes = []
                    anchor_nodes.extend(var_candidate_anchors.get(h_var, {}).get(int(hb), []))
                    for x in var_candidate_anchors.get(t_var, {}).get(int(tb), []):
                        if x not in anchor_nodes:
                            anchor_nodes.append(x)
                    if not anchor_nodes:
                        continue
                    new_qt = _instantiate_qt(head_val=entity_names[int(hb)], tail_val=entity_names[int(tb)])
                    out = run_constraint_retrieval_for_triple(
                        enc, reranker, entities, entity_names, triple_info, out_index, in_index,
                        question, new_qt, anchor_nodes, args, "second_pass", "double_var"
                    )
                    record["stage_records"].append(out["debug"])
                    if out["selected_triple_ids"]:
                        selected_all.extend(out["selected_triple_ids"])
                    if out["status"] == "resolved":
                        record["status"] = "resolved"

        selected_all = sorted(list(dict.fromkeys([int(x) for x in selected_all])))
        record["selected_triple_ids"] = selected_all
        if record.get("status") == "resolved" and selected_all:
            support[qi] = selected_all

        per_triple[qi] = record

    # Recompute bindings after propagation (one extra consolidation)
    bindings_raw2 = consolidate_bindings(var_to_induced_sets)
    bindings2 = {}
    for v, info in bindings_raw2.items():
        bset = info.get("binding", set()) or set()
        bindings2[v] = {
            "binding_ids": sorted([int(x) for x in bset]),
            "binding_names": [entity_names[int(x)] for x in sorted([int(x) for x in bset]) if 0 <= int(x) < len(entity_names)],
            "method": info.get("method"),
            "union_ids": sorted([int(x) for x in (info.get("union") or set())]),
            "intersection_ids": sorted([int(x) for x in (info.get("intersection") or set())]) if info.get("intersection") is not None else None,
        }

    # Evidence triples list
    evidence_ids = set()
    for tids in support.values():
        evidence_ids.update([int(t) for t in tids])

    evidence_triples = []
    idx_map = {}
    for k, tid in enumerate(sorted(evidence_ids)):
        info = triple_info[int(tid)]
        evidence_triples.append(
            {
                "kg_triple_id": int(tid),
                "head_entity_id": int(info["head_id"]) if isinstance(info.get("head_id"), int) else None,
                "tail_entity_id": int(info["tail_id"]) if isinstance(info.get("tail_id"), int) else None,
                "head": info.get("head"),
                "relation": info.get("relation"),
                "tail": info.get("tail"),
                "meta": info.get("meta") or {},
            }
        )
        idx_map[int(tid)] = int(k)

    support_list = []
    for qi, tids in support.items():
        ev_idx = [idx_map[int(tid)] for tid in tids if int(tid) in idx_map]
        if ev_idx:
            support_list.append({"query_triple_index": int(qi), "evidence_indices": ev_idx})

    debug = {
        "known_entities": known_entities,
        "entity_map": [
            {
                "query_entity": qe,
                "kg_entity_ids": [int(eid) for eid in emap[qe]],
                "kg_entity_names": [entity_names[eid] for eid in emap[qe]],
            }
            for qe in emap
        ],
        "bindings": bindings2,
        "per_triple": per_triple,
    }

    return {
        "id": item.get("id"),
        "question": item.get("question"),
        "ground_truth_answer": item.get("ground_truth_answer"),
        "query_plan": qtriples,
        "evidence_triples": evidence_triples,
        "support": support_list,
        "debug": debug,
    }
