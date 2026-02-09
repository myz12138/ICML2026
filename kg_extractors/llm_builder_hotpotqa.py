import argparse, os, re, json
from pathlib import Path
from collections import defaultdict

import torch

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.hotpot import load_hotpot, to_chunks
from llm_utils import LLMClient

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, *args, **kwargs):
        return x

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_BASE_URL"] = "https://hf-mirror.com"

def load_config(cfg_path: str):
    text = Path(cfg_path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Config file {cfg_path} is empty.")
    try:
        if cfg_path.endswith((".yaml", ".yml")):
            import yaml
            return yaml.safe_load(text)
        else:
            return json.loads(text)
    except Exception:
        try:
            import yaml
            return yaml.safe_load(text)
        except Exception as e:
            raise ValueError(f"Failed to parse config {cfg_path} as JSON or YAML: {e}")

def load_hotpot_json(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"HotpotQA file not found: {path}")

    try:
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            obj = obj["data"]
        if not isinstance(obj, list):
            raise ValueError(f"Expected a JSON list, got: {type(obj)}")
        return obj
    except json.JSONDecodeError:
        pass

    data = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def normalize(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()

def prompt_entities(chunk_text: str) -> list:
    sys_msg = "You extract canonical entities from text. Return concise JSON only."
    usr = f"""Text:
{chunk_text}

Extract entities as a JSON list. Each item:
{{
  "name": "...",
  "type": "PERSON|ORG|LOC|EVENT|OTHER",
  "aliases": ["...", "..."],
  "title_hint": "optional normalized title"
}}
Rules: deduplicate; be conservative; only entities supported by this text; 15 items max."""
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": usr},
    ]

def prompt_relations(text: str) -> list:
    sys_msg = "Extract factual relations present in the text. Output strict JSON list."
    usr = f"""Text:
{text}

Output JSON list of relations:
{{
  "head": "<entity name>",
  "relation": "<free-form predicate>",
  "tail": "<entity name>",
  "evidence": "<short quote from text>"
}}
Only include relations explicitly supported by text; 25 items max."""
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": usr},
    ]

def safe_json_parse(text: str):
    try:
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
        return json.loads(text)
    except Exception:
        return []

def llm_extract_entities(llm: LLMClient, chunk: str) -> list:
    out = llm.chat(prompt_entities(chunk), temperature=0.0, max_tokens=512)
    items = safe_json_parse(out)
    ents = []
    seen = set()
    for it in items if isinstance(items, list) else []:
        name = (it.get("name") or "").strip()
        if not name:
            continue
        key = normalize(name)
        if key in seen:
            continue
        seen.add(key)
        ents.append({
            "name": name,
            "type": it.get("type", "OTHER"),
            "title_hint": (it.get("title_hint") or "").strip()
        })
    return ents

def llm_extract_relations(llm: LLMClient, text: str) -> list:
    out = llm.chat(prompt_relations(text), temperature=0.0, max_tokens=1024)
    items = safe_json_parse(out)
    rels = []
    for it in items if isinstance(items, list) else []:
        h = (it.get("head") or "").strip()
        r = (it.get("relation") or "").strip()
        t = (it.get("tail") or "").strip()
        if h and r and t:
            rels.append({
                "head": h,
                "relation": r,
                "tail": t,
                "evidence": (it.get("evidence") or "").strip()
            })
    return rels

def append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def atomic_write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def safe_iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def build_llm_kg_jsonl_incremental(
    pages: dict,
    out_dir: str,
    num_pages: int = 2000,
    topk_chunks: int = 4,
    add_community: bool = True,
    community_algo: str = "label_prop",
    checkpoint_every_titles: int = 10,
    resume: bool = True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entities_path = out_dir / "entities.jsonl"
    triples_path = out_dir / "triples.jsonl"
    t2e_path = out_dir / "title2entities.jsonl"
    t2t_path = out_dir / "title2triples.jsonl"
    ckpt_path = out_dir / "checkpoint.json"

    titles = list(pages.keys())[:num_pages]
    llm = LLMClient()

    ckpt = load_json(ckpt_path, default={})
    start_title_idx = 0
    processed_titles = set()
    ent_key2id = {}
    entities_count = 0
    triples_count = 0
    triple_seen = set()

    if resume and ckpt.get("status") in ("running", "done") and entities_path.exists() and triples_path.exists():
        start_title_idx = int(ckpt.get("next_title_idx", 0))

        max_eid = -1
        for obj in safe_iter_jsonl(entities_path):
            if not isinstance(obj, dict):
                continue
            eid = obj.get("id")
            name = obj.get("name")
            if isinstance(eid, int) and isinstance(name, str):
                ent_key2id[normalize(name)] = eid
                if eid > max_eid:
                    max_eid = eid
        entities_count = max_eid + 1 if max_eid >= 0 else 0

        triples_count = 0
        for tr in safe_iter_jsonl(triples_path):
            if not isinstance(tr, dict):
                continue
            h_id = tr.get("h")
            t_id = tr.get("t")
            rel = tr.get("r")
            title = tr.get("title")
            if isinstance(h_id, int) and isinstance(t_id, int) and isinstance(rel, str) and isinstance(title, str):
                rel_norm = normalize(rel)
                triple_seen.add((h_id, t_id, rel_norm, title))
                triples_count += 1

        for obj in safe_iter_jsonl(t2t_path):
            if isinstance(obj, dict) and isinstance(obj.get("title"), str):
                processed_titles.add(obj["title"])

    else:
        entities_path.write_text("", encoding="utf-8")
        triples_path.write_text("", encoding="utf-8")
        t2e_path.write_text("", encoding="utf-8")
        t2t_path.write_text("", encoding="utf-8")
        atomic_write_json(ckpt_path, {"status": "running", "next_title_idx": 0})

    def get_or_create_entity(name: str, etype: str = "OTHER"):
        nonlocal entities_count
        key = normalize(name)
        if not key:
            return None
        if key in ent_key2id:
            return ent_key2id[key]
        i = entities_count
        entities_count += 1
        append_jsonl(entities_path, {
            "id": i,
            "name": name,
            "type": etype
        })
        ent_key2id[key] = i
        return i

    for idx in tqdm(range(start_title_idx, len(titles)), desc="Building LLM KG (pages)"):
        t = titles[idx]
        if t in processed_titles:
            continue

        text = pages[t]
        chunks, _ = to_chunks({t: text}, max_chars=600)
        if not chunks:
            if (idx + 1) % checkpoint_every_titles == 0:
                atomic_write_json(ckpt_path, {
                    "status": "running",
                    "next_title_idx": idx + 1,
                    "entities_count": entities_count,
                    "triples_count": triples_count
                })
            continue

        title_entities_set = set()
        title_triples_list = []

        ents_local = {}
        for c in chunks:
            ents = llm_extract_entities(llm, c)
            for e in ents:
                key = normalize(e["name"])
                if key not in ents_local:
                    ents_local[key] = e

        for key, e in ents_local.items():
            eid = get_or_create_entity(e["name"], e.get("type", "OTHER"))
            if eid is not None:
                title_entities_set.add(eid)

        for c in chunks:
            rels = llm_extract_relations(llm, c)
            for r in rels:
                h_id = get_or_create_entity(r["head"])
                t_id = get_or_create_entity(r["tail"])
                if h_id is None or t_id is None:
                    continue
                rel_norm = normalize(r["relation"])
                key_tr = (h_id, t_id, rel_norm, t)
                if key_tr in triple_seen:
                    continue
                triple_seen.add(key_tr)

                triple_idx = triples_count
                triples_count += 1

                triple_obj = {
                    "h": h_id,
                    "t": t_id,
                    "r": r["relation"],
                    "evidence": r.get("evidence", ""),
                    "title": t
                }
                append_jsonl(triples_path, triple_obj)

                title_triples_list.append(triple_idx)
                title_entities_set.add(h_id)
                title_entities_set.add(t_id)

        append_jsonl(t2e_path, {"title": t, "entity_ids": sorted(list(title_entities_set))})
        append_jsonl(t2t_path, {"title": t, "triple_idxs": title_triples_list})
        processed_titles.add(t)

        if (idx + 1) % checkpoint_every_titles == 0:
            atomic_write_json(ckpt_path, {
                "status": "running",
                "next_title_idx": idx + 1,
                "entities_count": entities_count,
                "triples_count": triples_count
            })

    atomic_write_json(ckpt_path, {
        "status": "done",
        "next_title_idx": len(titles),
        "entities_count": entities_count,
        "triples_count": triples_count,
        "out_dir": str(out_dir)
    })

    print(f"[jsonl_builder] Saved (incremental JSONL, no A) to {out_dir}")
    print(f"[jsonl_builder] entities={entities_count}, triples={triples_count}")
    print(f"[jsonl_builder] files:")
    print(f"  - {entities_path}")
    print(f"  - {triples_path}")
    print(f"  - {t2e_path}")
    print(f"  - {t2t_path}")
    print(f"  - {ckpt_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default='datasets/dataset/hotpotqa.json',
                    help="Path to a HotpotQA json/jsonl file. If set, overrides --config/--split.")
    ap.add_argument("--config", default="./configs/hotpot_local.yaml")
    ap.add_argument("--split", choices=["train", "dev"], default="dev")

    ap.add_argument("--num_examples", type=int, default=1000,
                    help="How many Hotpot examples to use as seed corpus.")
    ap.add_argument("--num_pages", type=int, default=None,
                    help="(deprecated) kept for compatibility; use --num_examples instead.")

    ap.add_argument("--topk_chunks", type=int, default=4)
    ap.add_argument("--no_community", action="store_true",
                    help="(kept for compatibility; ignored)")
    ap.add_argument("--out", default="./kg_extractors/seedkg_hotpotqa_1000_jsonl_2step_new",
                    help="Output directory for JSONL files (entities/triples/mappings).")

    ap.add_argument("--checkpoint_every_titles", type=int, default=10,
                    help="Write checkpoint.json every N titles.")
    ap.add_argument("--no_resume", action="store_true",
                    help="Disable resume; overwrite existing outputs in --out directory.")

    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.data is not None:
        data = load_hotpot_json(args.data)
        print(f"[llm_builder] loaded HotpotQA from --data: {args.data} (n={len(data)})")
    

    if args.num_pages is not None:
        num_examples = args.num_pages
    else:
        num_examples = args.num_examples

    pages = {}
    used_ex = 0
    for ex in data:
        if used_ex >= num_examples:
            break
        for title, sents in ex["context"]:
            if title not in pages:
                pages[title] = " ".join(sents)
        used_ex += 1

    print(f"[llm_builder] use {used_ex} examples, collected {len(pages)} unique pages for KG extraction.")

    build_llm_kg_jsonl_incremental(
        pages,
        out_dir=args.out,
        num_pages=len(pages),
        topk_chunks=args.topk_chunks,
        add_community=(not args.no_community),
        community_algo="label_prop",
        checkpoint_every_titles=args.checkpoint_every_titles,
        resume=(not args.no_resume),
    )

if __name__ == "__main__":
    main()