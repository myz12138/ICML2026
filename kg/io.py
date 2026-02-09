
import json
import os

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, obj, indent=2):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)

def load_list_json(path):
    data = load_json(path)
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list (or a dict with a 'data' list).")
    return data

def load_raw_dataset_id2qa(dataset, path):
    if not path or not os.path.exists(path):
        return {}
    data = load_list_json(path)
    id2qa = {}
    for ex in data:
        if not isinstance(ex, dict):
            continue
        ex_id = ex.get("id") or ex.get("_id") or ex.get("qid")
        if ex_id is None and dataset == "musique":
            ex_id = ex.get("id")
        if ex_id is None:
            continue
        q = ex.get("question")
        ans = ex.get("answer")
        id2qa[str(ex_id)] = {"question": q, "answer": ans}
    return id2qa

def normalize_query_triple(qt):
    if not isinstance(qt, dict):
        return {"head": "", "tail": "", "relation_variants": []}
    qt = dict(qt)
    qt.pop("relation", None)  # query-side relation removed
    h = qt.get("head", "")
    t = qt.get("tail", "")
    rv = qt.get("relation_variants", [])
    if not isinstance(rv, list):
        rv = []
    rv2, seen = [], set()
    for x in rv:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if not s or s.lower() in seen:
            continue
        seen.add(s.lower())
        rv2.append(s)
    return {"head": h, "tail": t, "relation_variants": rv2}

def normalize_query_items(dataset, query_items, id2qa=None):
    id2qa = id2qa or {}
    out = []
    for it in query_items:
        if not isinstance(it, dict):
            continue
        qid = it.get("id") or it.get("_id") or it.get("qid")
        if qid is None:
            continue
        qid = str(qid)

        q = it.get("question") or (id2qa.get(qid, {}) or {}).get("question")
        ans = it.get("ground_truth_answer")
        if ans is None:
            ans = it.get("answer")
        if ans is None:
            ans = (id2qa.get(qid, {}) or {}).get("answer")

        qp = it.get("query_plan") or it.get("query_triples") or it.get("triples")
        if qp is None:
            qg = it.get("query_graph") or {}
            if isinstance(qg, dict):
                qp = qg.get("query_plan") or qg.get("triples")
        if not isinstance(qp, list):
            raise ValueError(f"Item {qid} has no query_plan/triples list.")

        qp2 = [normalize_query_triple(x) for x in qp]

        out.append(
            {"id": qid, "question": q, "ground_truth_answer": ans, "query_plan": qp2}
        )
    return out
