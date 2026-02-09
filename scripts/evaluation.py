import json
import os
import re
import string
from tqdm import tqdm
from openai import OpenAI
from ..configs.config import parse_args


def build_client(args):
    return OpenAI(api_key=args.api_key, base_url=args.base_url)


def normalize_whitespace(text):
    return " ".join((text or "").split())


def normalize_answer(s):
    def lower(text):
        return text.lower()

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s or ""))))


def exact_match_score(prediction, ground_truth):
    if not ground_truth:
        return 0.0
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    if not ground_truth:
        return 0.0
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in gold_tokens:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1

    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def load_list_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError("DATA_JSON must be a JSON list (or a dict with a 'data' list).")
    return data


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


def gold_support_set(dataset, ex):
    """
    2wiki/hotpotqa: sentence-level supports => set of (title, sent_idx)
    musique: paragraph-level supports => set of para_idx (int)
    """
    if ex is None:
        return set()

    if dataset in ("2wiki", "hotpotqa"):
        gold = set()
        for sf in ex.get("supporting_facts", []) or []:
            title = normalize_whitespace(str(sf[0] or "")).strip()
            try:
                si = int(sf[1])
            except Exception:
                continue
            gold.add((title, si))
        return gold

    gold = set()
    for step in ex.get("question_decomposition", []) or []:
        if not isinstance(step, dict):
            continue
        if "paragraph_support_idx" in step:
            try:
                gold.add(int(step["paragraph_support_idx"]))
            except Exception:
                pass
    return gold


def _format_triple(t):
    """
    - For KG triples: includes relation.
    - For query triples (no canonical relation): use relation_variants if available.
    """
    if isinstance(t, dict):
        h = str(t.get("head", "") or "").strip()
        ta = str(t.get("tail", "") or "").strip()

        r = str(t.get("relation", "") or "").strip()
        if not r:
            rv = t.get("relation_variants", [])
            if isinstance(rv, list) and rv:
                r = " / ".join([str(x).strip() for x in rv[:3] if str(x).strip()])
        parts = [x for x in [h, r, ta] if x]
        return " | ".join(parts) if parts else ""
    if isinstance(t, (list, tuple)) and len(t) >= 3:
        h = str(t[0] or "").strip()
        r = str(t[1] or "").strip()
        ta = str(t[2] or "").strip()
        parts = [x for x in [h, r, ta] if x]
        return " | ".join(parts) if parts else ""
    return str(t or "").strip()


def build_prompt_from_stage2_item(item, args):
    question = str(item.get("question", "") or "").strip()
    triples_evidence = item["triples_evidence"] 

    intro = (
        "You are a strict multi hop inference assistant that can complete multi hop inference based on given information. "
        "Now, you need to conduct rigorous and rational multi-step thinking based on the given search evidence to answer this question. "
        "You only need to use the given information to answer the question. "
        "We decomposed the problem and generated multiple subqueries, and retrieved several relevant evidence for each subquery for you to think and reason about the answer step by step. "
        "You need to choose the most suitable evidence content as much as possible to solve this unknown tuple, and further deduce the next unknown tuple based on the solved content until you answer the question. "
        "You can only answer the final answer with short and appropriate phrases (such as names, numbers, or short noun phrases that conform to the question format). "
        "Do not include explanations, sentences, or any other words. "
        "Here is the information you can obtain:"
    )

    ev_lines = []
    for te in triples_evidence[: args.prompt_max_triples]:
        qi = int(te["query_triple_index"])
        qt = _format_triple(te["query_triple"])
        ev_lines.append(f"T{qi}: {qt}")
        cand = te.get("candidate_evidences") or []
        if isinstance(cand, list) and cand:
            ev_lines.append("  Text evidence:")
            for i, ev in enumerate(cand[: args.prompt_max_text_per_triple], start=1):
                ctx = normalize_whitespace(ev.get("context") or ev.get("evidence") or "").strip()
                if ctx:
                    ev_lines.append(f"    Evidence {i}: {ctx}")


    ev_block = "\n".join(ev_lines)

    body = (
        f"Question: {question}\n\n"
        "The query tuple after problem decomposition is as follows: "
        "(where the relationship is an approximate query relationship related to the semantics of the problem, "
        "and the entity is an unknown entity type that needs to be solved based on the provided evidence, not the true answer).\n"
        f"For each decomposed query triple, the retrieved evidence is as follows:\n{ev_block}\n\n"
        "Based on the above information, answer the question.\n\n"
        f"Question: {question} Answer:"
    )

    return intro + "\n" + body


def find_best_sent_for_evidence_2wiki_like(ex, title, ev_text):
    """
    evidence text -> best (title, sent_idx) in ex["context"] by substring + token-F1.
    Returns (normalized_title, sent_idx, score) or None.
    """
    ctx = ex.get("context") or ex.get("contexts") or ex.get("ctxs") or []
    title = (title or "").strip()
    ev_text = (ev_text or "").strip()
    ev_norm = normalize_answer(ev_text)

    cands = []
    for block in ctx:
        blk_title = ""
        sents = None

        if isinstance(block, list) and len(block) >= 2:
            blk_title = str(block[0] or "").strip()
            sents = block[1]
        elif isinstance(block, dict):
            blk_title = str(block.get("title") or "").strip()
            sents = block.get("sentences") or block.get("sents") or block.get("text") or block.get("context")
        else:
            continue

        if isinstance(sents, str):
            sents = [sents]
        if not isinstance(sents, list):
            continue

        if title and blk_title and blk_title != title:
            continue

        for si, s in enumerate(sents):
            if isinstance(s, list):
                s = " ".join([str(x) for x in s if x is not None])
            s = str(s or "").strip()
            if s:
                cands.append((blk_title, si, s))

    best = None
    best_score = -1.0
    for blk_title, si, s in cands:
        score = 0.0
        s_norm = normalize_answer(s)
        if ev_norm and s_norm and (ev_norm in s_norm or s_norm in ev_norm):
            score += 1.0
        score += f1_score(ev_text, s) if ev_text else 0.0
        if score > best_score:
            best_score = score
            best = (normalize_whitespace(blk_title).strip(), int(si), float(score))

    return best if best_score > 0.0 else None


def find_best_para_for_evidence_musique(ex, title, ev_text):
    ps = ex.get("paragraphs", []) or []
    title = (title or "").strip()
    ev_text = (ev_text or "").strip()
    ev_norm = normalize_answer(ev_text)

    cands = []
    if title:
        for p in ps:
            if isinstance(p, dict) and (str(p.get("title") or "").strip() == title):
                cands.append(p)
    if not cands:
        cands = [p for p in ps if isinstance(p, dict)]

    best_idx = None
    best_score = -1.0
    for p in cands:
        txt = (p.get("paragraph_text") or "").strip()
        if not txt:
            continue
        score = 0.0
        txt_norm = normalize_answer(txt)
        if ev_norm and txt_norm and (ev_norm in txt_norm or txt_norm in ev_norm):
            score += 1.0
        score += f1_score(ev_text, txt) if ev_text else 0.0
        if score > best_score:
            best_score = score
            try:
                best_idx = int(p.get("idx"))
            except Exception:
                best_idx = None
    return (best_idx, float(best_score)) if best_score > 0.0 else None


def build_predicted_units(dataset, item, id2ex):

    qid = str(item.get("id"))
    ex = id2ex.get(qid)
    if ex is None:
        return []

    units = []
    seen = set()

    for te in item["triples_evidence"]:
        for ev in (te.get("candidate_evidences") or []):
            title = (ev.get("title") or "").strip()
            ev_text = ev.get("evidence") or ev.get("context") or ""
            sim = float(ev.get("similarity", 0.0) or 0.0)

            if dataset in ("2wiki", "hotpotqa"):
                mapped = None
                if ev.get("sent_idx", None) is not None:
                    mapped = (normalize_whitespace(title).strip(), int(ev["sent_idx"]), sim)
                else:
                    mp = find_best_sent_for_evidence_2wiki_like(ex, title, ev_text)
                    if mp is not None:
                        mapped = (mp[0], mp[1], sim)
                if mapped is None:
                    continue
                key = (mapped[0], mapped[1])
                if key in seen:
                    continue
                seen.add(key)
                units.append(mapped)
            else:
                mapped = None
                if ev.get("paragraph_idx", None) is not None:
                    mapped = (int(ev["paragraph_idx"]), sim)
                else:
                    mp = find_best_para_for_evidence_musique(ex, title, ev_text)
                    if mp is not None:
                        mapped = (mp[0], sim)
                if mapped is None:
                    continue
                key = mapped[0]
                if key in seen:
                    continue
                seen.add(key)
                units.append(mapped)

    return units



def compute_recall(dataset, item, id2ex, predicted_units, recall_k_list):
    qid = str(item.get("id"))
    ex = id2ex.get(qid)
    gold = gold_support_set(dataset, ex)
    if not gold or not predicted_units:
        return {}, {}, 0

    total_gold = len(gold)
    hits_k = {}
    recalls = {}

    if dataset in ("2wiki", "hotpotqa"):
        pred_keys = [(t, si) for (t, si, _sc) in predicted_units]
    else:
        pred_keys = [pid for (pid, _sc) in predicted_units]

    for k in recall_k_list:
        k_val = min(k, len(pred_keys))
        topk = set(pred_keys[:k_val])
        hits = len(gold & topk)
        hits_k[f"hits@{k}"] = hits
        recalls[f"recall@{k}"] = hits / total_gold if total_gold else 0.0

    hits_all = len(gold & set(pred_keys))
    hits_k["hits@all"] = hits_all
    recalls["recall_all"] = hits_all / total_gold if total_gold else 0.0
    return recalls, hits_k, total_gold


def _parse_recall_k_list(s):
    out = []
    for x in str(s or "").split(","):
        x = x.strip()
        if not x:
            continue
        try:
            out.append(int(x))
        except Exception:
            pass
    return out if out else [2, 5, 10]



def main_ev():
    args = parse_args()

    dataset = args.dataset
    recall_k_list = _parse_recall_k_list(args.recall_k_list)

    client = build_client(args)
    id2ex = load_id2ex(dataset, args.data_json)

    with open(args.stage2_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    if args.num_samples is not None:
        items = items[: int(args.num_samples)]

    outputs = []
    total_em = 0.0
    total_f1 = 0.0
    n = 0

    global_hits = {k: 0 for k in recall_k_list}
    global_hits["all"] = 0
    global_total_gold = 0
    avg_recall_sum = {k: 0.0 for k in recall_k_list}
    avg_recall_sum["all"] = 0.0
    num_q_with_gold = 0

    for item in tqdm(items, desc=f"Evaluation universal ({dataset})"):
        question = item.get("question", "")
        gold_ans = item.get("ground_truth_answer", item.get("answer", ""))

        prompt = build_prompt_from_stage2_item(item, args)

        resp = client.chat.completions.create(
            model=args.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful question answering assistant. "
                        "Always ground your answers strictly in the provided evidences, "
                        "and do not use outside knowledge."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        pred = (resp.choices[0].message.content or "").strip()

        em = exact_match_score(pred, gold_ans)
        f1 = f1_score(pred, gold_ans)
        total_em += em
        total_f1 += f1
        n += 1

        predicted_units = build_predicted_units(dataset, item, id2ex)
        recalls, hits_k, total_gold = compute_recall(dataset, item, id2ex, predicted_units, recall_k_list)

        if total_gold > 0:
            num_q_with_gold += 1
            global_total_gold += total_gold
            for k in recall_k_list:
                global_hits[k] += hits_k.get(f"hits@{k}", 0)
                avg_recall_sum[k] += recalls.get(f"recall@{k}", 0.0)
            global_hits["all"] += hits_k.get("hits@all", 0)
            avg_recall_sum["all"] += recalls.get("recall_all", 0.0)

        out_item = {
            "id": item.get("id"),
            "question": question,
            "ground_truth_answer": gold_ans,
            "model_input": prompt,
            "model_answer": pred,
            "em": em,
            "f1": f1,
            "retrieval_predicted_support_units": predicted_units,
            "retrieval_total_support_facts": total_gold,
        }
        for k in recall_k_list:
            out_item[f"retrieval_recall@{k}"] = recalls.get(f"recall@{k}", 0.0)
        out_item["retrieval_recall_all"] = recalls.get("recall_all", 0.0)
        outputs.append(out_item)

    overall_em = total_em / n if n else 0.0
    overall_f1 = total_f1 / n if n else 0.0

    print(f"Overall EM: {overall_em:.4f}")
    print(f"Overall F1: {overall_f1:.4f}")

    if global_total_gold and num_q_with_gold:
        for k in recall_k_list:
            print(f"Global retrieval recall@{k}: {global_hits[k] / global_total_gold:.4f}")
        print(f"Global retrieval recall_all: {global_hits['all'] / global_total_gold:.4f}")
        for k in recall_k_list:
            print(f"Avg retrieval recall@{k} per question: {avg_recall_sum[k] / num_q_with_gold:.4f}")
        print(f"Avg retrieval recall_all per question: {avg_recall_sum['all'] / num_q_with_gold:.4f}")

    out_dir = os.path.dirname(args.eval_output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.eval_output_json, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main_ev()
