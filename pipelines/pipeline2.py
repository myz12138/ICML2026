
from ..io import load_list_json
from ..sim.embedding import build_embedder
from ..rerank.cross_encoder import build_reranker

from .textual_recovery import retrieve_2wiki_like, retrieve_musique
from .provenance import tri_in_text
from .dataset_adapters import iter_context_units_2wiki_like, iter_paragraphs_musique

def load_id2ex(dataset, path):
    data = load_list_json(path)
    id2ex = {}
    for ex in data:
        if not isinstance(ex, dict):
            continue
        ex_id = ex.get("_id") or ex.get("id") or ex.get("qid")
        if ex_id is None and dataset == "musique":
            ex_id = ex.get("id")
        if ex_id is None:
            continue
        id2ex[str(ex_id)] = ex
    return id2ex

def build_support_map(stage1_item):
    ev_triples = stage1_item.get("evidence_triples") or []
    support = stage1_item.get("support") or []
    mp = {}
    if not isinstance(ev_triples, list) or not isinstance(support, list):
        return mp
    for s in support:
        if not isinstance(s, dict):
            continue
        qi = s.get("query_triple_index", None)
        idxs = s.get("evidence_indices", []) or []
        try:
            qi = int(qi)
        except Exception:
            continue
        out = []
        if isinstance(idxs, list):
            for ix in idxs:
                try:
                    j = int(ix)
                except Exception:
                    continue
                if 0 <= j < len(ev_triples) and isinstance(ev_triples[j], dict):
                    out.append(ev_triples[j])
        if out:
            mp[qi] = out
    return mp

def resolved_sentence_evidence_2wiki_like(ex, kg_triple):
    title = (kg_triple.get("meta") or {}).get("title") or ""
    title = str(title or "").strip()
    tri = {"head": kg_triple.get("head"), "relation": kg_triple.get("relation"), "tail": kg_triple.get("tail")}

    for doc_idx, t, sents in iter_context_units_2wiki_like(ex):
        if title and t != title:
            continue
        for si, sent in enumerate(sents):
            if tri_in_text(sent, tri, require_rel=False):
                return {
                    "title": t,
                    "paragraph_idx": int(doc_idx),
                    "sent_idx": int(si),
                    "context": f"{t} {sent}".strip(),
                    "evidence": sent,
                    "similarity": 1.0,
                    "sentence_similarity": 1.0,
                }
        if title and t == title:
            break

    if title:
        for doc_idx, t, sents in iter_context_units_2wiki_like(ex):
            if t != title:
                continue
            if sents:
                sent = sents[0]
                return {
                    "title": t,
                    "paragraph_idx": int(doc_idx),
                    "sent_idx": 0,
                    "context": f"{t} {sent}".strip(),
                    "evidence": sent,
                    "similarity": 0.0,
                    "sentence_similarity": 0.0,
                }
    return None

def resolved_paragraph_evidence_musique(ex, kg_triple):
    title = (kg_triple.get("meta") or {}).get("title") or ""
    title = str(title or "").strip()
    tri = {"head": kg_triple.get("head"), "relation": kg_triple.get("relation"), "tail": kg_triple.get("tail")}
    for para_idx, t, txt in iter_paragraphs_musique(ex):
        if title and t != title:
            continue
        if tri_in_text(txt, tri, require_rel=False):
            return {
                "title": t,
                "paragraph_idx": int(para_idx),
                "sent_idx": 0,
                "context": txt,
                "evidence": "",
                "similarity": 1.0,
                "sentence_similarity": 1.0,
            }
    return None

def process_item_stage2(dataset, stage1_item, ex, enc, reranker, args):
    question = stage1_item.get("question", "")
    query_triples = stage1_item.get("query_plan") or []
    debug = stage1_item.get("debug") or {}
    bindings = (debug.get("bindings") or {})
    entity_map = debug.get("entity_map") or []

    support_map = build_support_map(stage1_item)

    triples_evidence = []
    for qi, qt in enumerate(query_triples):
        status = "unresolved"
        per_tr = (debug.get("per_triple") or [])
        if isinstance(per_tr, list) and qi < len(per_tr) and isinstance(per_tr[qi], dict):
            status = per_tr[qi].get("status") or status

        candidate_evidences = []
        kg_triples = []

        if status == "resolved":
            kg_triples = support_map.get(int(qi), []) or []
            for tri in kg_triples:
                if dataset in ("2wiki", "hotpotqa"):
                    ev = resolved_sentence_evidence_2wiki_like(ex, tri) if ex is not None else None
                else:
                    ev = resolved_paragraph_evidence_musique(ex, tri) if ex is not None else None
                if ev:
                    candidate_evidences.append(ev)
            candidate_evidences = candidate_evidences[: max(1, int(args.topk_resolved))]
        else:
            if ex is not None:
                if dataset in ("2wiki", "hotpotqa"):
                    candidate_evidences = retrieve_2wiki_like(
                        enc, reranker, ex, question, qt, bindings, entity_map,
                        topk=args.topk_unresolved,
                        doc_prefilter_k=args.doc_prefilter_k,
                        unres_top_similar=args.unres_top_similar,
                        max_length=args.rerank_max_length,
                        batch_size=args.rerank_batch,
                    )
                else:
                    candidate_evidences = retrieve_musique(
                        enc, reranker, ex, question, qt, bindings, entity_map,
                        topk=args.topk_unresolved,
                        unres_top_similar=args.unres_top_similar,
                        max_length=args.rerank_max_length,
                        batch_size=args.rerank_batch,
                    )

        triples_evidence.append(
            {
                "query_triple_index": int(qi),
                "query_triple": qt,
                "triple_status": status,
                "candidate_evidences": candidate_evidences,
                "kg_triples": kg_triples,
            }
        )

    return {
        "id": str(stage1_item.get("id")),
        "question": question,
        "ground_truth_answer": stage1_item.get("ground_truth_answer", ""),
        "triples_evidence": triples_evidence,
    }

def run_stage2(dataset, stage1_items, data_json, args):
    id2ex = load_id2ex(dataset, data_json)
    enc = build_embedder(args.emb_model, args.emb_device)
    reranker = build_reranker(args.rerank_model, fp16=bool(args.rerank_fp16))

    results = []
    for it in stage1_items:
        qid = str(it.get("id"))
        ex = id2ex.get(qid)
        results.append(process_item_stage2(dataset, it, ex, enc, reranker, args))
    return results