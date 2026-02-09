
import json, os, time
import argparse
from tqdm import tqdm
from openai import OpenAI

SYSTEM_PROMPT = """You are an expert in Knowledge Graphs and Question Answering. 
Your task is to decompose a complex natural language question into a sequence of triples.

Rules:
1. Identify known entities and use '?'-prefixed variables for unknown ones.
2. Reuse the same variable only when it clearly refers to the same unknown.
3. Output a valid JSON object with one key: "triples".
4. Each element in "triples" must be an object containing:
   - "head": string
   - "relation_variants": 3 short, logically similar relation strings
   - "tail": string
5. Triples must form a minimal yet sufficient reasoning chain: filling all "?" enables answering the query using only these triples.
6. For property queries (nationality/position/time/location/etc.) link the property "?" directly to the target entity variable, not a side entity.
7. Make "relation_variants"informative enough to be understood without the original query.
8. Encode extra conditions as constraints on the same entity variable via additional triples; don't drop useful descriptors.
9. Never let one variable represent multiple distinct unknowns; use separate names when unclear. (e.g. ?country1, ?country2) when needed.
10. If the query provides a concrete value (country/year/location), connect directly to it—don't introduce a ? just to equal that value.
11. Keep variable names short/generic; express details via relations/triples, not long variable names.


### EXAMPLES

**Input**: "The Oberoi family is part of a hotel company that has a head office in what city?"
**Output**:
{
  "triples": [
    {
      "head": "Oberoi family",
      "relation_variants": ["part of", "is part of", "belongs to", "is a member of"],
      "tail": "?hotel company"
    },
    {
      "head": "?hotel company",
      "relation_variants": ["has", "has as a feature", "possesses"],
      "tail": "a head office"
    },
    {
      "head": "?hotel company",
      "relation_variants": ["located in", "is based in", "is situated in"],
      "tail": "?city"
    }
  ]
}

**Input**: "Which film starring Tom Hanks was directed by Steven Spielberg and released in 1998?"
**Output**:
{
  "triples": [
    {
      "head": "Tom Hanks",
      "relation_variants": ["starred in", "appeared in", "acted in"],
      "tail": "?film"
    },
    {
      "head": "?film",
      "relation_variants": ["directed by", "was directed by", "film direction by"],
      "tail": "Steven Spielberg"
    },
    {
      "head": "?film",
      "relation_variants": ["publication date", "release year", "released in"],
      "tail": "1998"
    }
  ]
}
"""

def load_dataset(path, num_samples):
    data = json.loads(open(path, "r", encoding="utf-8").read())
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list.")
    if num_samples is None:
        return data
    return data[:int(num_samples)]

def get_ex_id(ex):
    return str(ex.get("id") or ex.get("_id") or ex.get("qid") or ex.get("question_id") or "")

def get_question(ex):
    return ex.get("question", "")

def get_answer(ex):
    return ex.get("answer", "")


def _normalize_triples(triples_raw):
    normalized = []
    if not isinstance(triples_raw, list):
        return normalized

    for tr in triples_raw:
        if not isinstance(tr, dict):
            continue

        head = "" if tr.get("head") is None else str(tr.get("head"))
        tail = "" if tr.get("tail") is None else str(tr.get("tail"))

        rv = tr.get("relation_variants")
        rv_clean = []
        if isinstance(rv, list):
            seen = set()
            for x in rv:
                if not isinstance(x, str):
                    continue
                s = x.strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                rv_clean.append(s)

        normalized.append(
            {
                "head": head.strip(),
                "relation_variants": rv_clean,
                "tail": tail.strip(),
            }
        )

    return normalized


def get_query_plan_with_retry(question, client, model_name, max_retries=3):
    user_content = f"Question: {question}\nOutput:"
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=30.0,
            )
            content = resp.choices[0].message.content
            data = json.loads(content)

            triples_raw = None
            if isinstance(data, dict) and "triples" in data and isinstance(data["triples"], list):
                triples_raw = data["triples"]
            elif isinstance(data, dict):
                # fallback: first list value
                for _, v in data.items():
                    if isinstance(v, list):
                        triples_raw = v
                        break

            if triples_raw is None:
                return []

            return _normalize_triples(triples_raw)

        except Exception as e:
            print(f"[Warn] LLM failed (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2)

    return []


def main_query(args):
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    data = load_dataset(args.dataset, args.data_json, args.num_samples)
    print(f"[QueryBuilder] dataset={args.dataset} examples={len(data)}")
    print(f"[QueryBuilder] output={args.query_json}")

    out_dir = os.path.dirname(args.query_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.query_json, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True

        for ex in tqdm(data):
            qid = get_ex_id(ex)
            question = get_question(ex)
            answer = get_answer(ex)

            triples = get_query_plan_with_retry(question, client, args.model_name)

            item = {
                "id": qid,
                "question": question,
                "ground_truth_answer": answer,
                "query_plan": triples,
            }

            if not first:
                f.write(",\n")
            first = False

            json.dump(item, f, ensure_ascii=False, indent=2)
            f.flush()

            if args.sleep_secs:
                time.sleep(args.sleep_secs)

        f.write("\n]\n")

    print("[QueryBuilder] done.")

