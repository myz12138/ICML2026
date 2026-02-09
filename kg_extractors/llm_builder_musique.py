import argparse
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from data.hotpot import to_chunks as repo_to_chunks
except Exception:
    repo_to_chunks = None

from utils.llm import LLMClient

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, *args, **kwargs):
        return x

_ALLOWED_TYPES = {"PERSON", "ORG", "LOC", "EVENT", "OTHER"}

def normalize(s: str) -> str:
    return re.sub(r"\W+", " ", (s or "").lower()).strip()

def normalize_type(t: str) -> str:
    t0 = (t or "").strip()
    if not t0:
        return "OTHER"
    up = t0.upper()
    if up in _ALLOWED_TYPES:
        return up
    if "PER" in up or up == "PERSON":
        return "PERSON"
    if "ORG" in up or "COMP" in up or "GPE" in up:
        return "ORG" if "GPE" not in up else "LOC"
    if "LOC" in up or "PLACE" in up or "CITY" in up or "COUNTRY" in up or "GPE" in up:
        return "LOC"
    if "EVENT" in up:
        return "EVENT"
    return "OTHER"

def safe_json_parse_any(text: str) -> Any:
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None

    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    try:
        return json.loads(s)
    except Exception:
        pass

    try:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end + 1])
    except Exception:
        pass

    try:
        start = s.find("[")
        end = s.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end + 1])
    except Exception:
        pass

    return None

def prompt_openie(chunk_text: str) -> List[Dict[str, str]]:
    sys_msg = (
        "You are an expert at Open Information Extraction (OpenIE) for knowledge graphs. "
        "Extract named entities and factual triples supported by the passage. "
        "Resolve coreference (pronouns) to specific names. "
        "Return STRICT JSON ONLY (no markdown)."
    )
    usr = f"""Passage:
\"\"\"{chunk_text}\"\"\"

Return a JSON object with this schema:
{{
  "entities": [
    {{"name": "…", "type": "PERSON|ORG|LOC|EVENT|OTHER"}}
  ],
  "triples": [
    {{"head": "…", "relation": "…", "tail": "…", "evidence": "short quote"}}
  ]
}}

Constraints:
- Only include entities/relations explicitly supported by the passage.
- Keep entities concise and canonical (deduplicate).
- "evidence" should be a short substring from the passage (<= 20 words).
- Max 20 entities, max 30 triples.
"""
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": usr}]

def llm_openie_extract(llm: LLMClient, chunk_text: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    raw = llm.chat(prompt_openie(chunk_text), temperature=0.0, max_tokens=1024)
    parsed = safe_json_parse_any(raw)

    entities: List[Dict[str, str]] = []
    triples: List[Dict[str, str]] = []

    if parsed is None:
        return entities, triples

    if isinstance(parsed, list):
        parsed = {"triples": parsed}

    if not isinstance(parsed, dict):
        return entities, triples

    ent_items = parsed.get("entities", None)
    if ent_items is None:
        ent_items = parsed.get("named_entities", None)
    if isinstance(ent_items, list):
        for it in ent_items:
            if isinstance(it, str):
                name = it.strip()
                if name:
                    entities.append({"name": name, "type": "OTHER"})
            elif isinstance(it, dict):
                name = (it.get("name") or it.get("text") or "").strip()
                if not name:
                    continue
                etype = normalize_type(it.get("type") or "OTHER")
                entities.append({"name": name, "type": etype})

    tri_items = parsed.get("triples", None)
    if isinstance(tri_items, list):
        for it in tri_items:
            if isinstance(it, dict):
                h = (it.get("head") or it.get("h") or "").strip()
                r = (it.get("relation") or it.get("r") or "").strip()
                t = (it.get("tail") or it.get("t") or "").strip()
                if h and r and t:
                    triples.append({
                        "head": h,
                        "relation": r,
                        "tail": t,
                        "evidence": (it.get("evidence") or "").strip()
                    })
            elif isinstance(it, list) and len(it) >= 3:
                h, r, t = it[0], it[1], it[2]
                if isinstance(h, str) and isinstance(r, str) and isinstance(t, str):
                    triples.append({
                        "head": h.strip(),
                        "relation": r.strip(),
                        "tail": t.strip(),
                        "evidence": ""
                    })

    dedup_ents: Dict[str, Dict[str, str]] = {}
    for e in entities:
        k = normalize(e.get("name", ""))
        if not k:
            continue
        if k not in dedup_ents:
            dedup_ents[k] = {"name": e["name"], "type": normalize_type(e.get("type", "OTHER"))}

    entities = list(dedup_ents.values())
    return entities, triples

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

def fallback_to_chunks(title2text: Dict[str, str], max_chars: int) -> Tuple[List[str], List[str]]:
    chunks: List[str] = []
    titles: List[str] = []
    for title, text in title2text.items():
        if not text:
            continue
        paras = re.split(r"\n{2,}", text)
        buf = ""
        for p in paras:
            p = p.strip()
            if not p:
                continue
            if not buf:
                buf = p
            elif len(buf) + 2 + len(p) <= max_chars:
                buf = buf + "\n\n" + p
            else:
                chunks.append(buf)
                titles.append(title)
                buf = p
        if buf:
            chunks.append(buf)
            titles.append(title)
    return chunks, titles

def make_chunks(title: str, text: str, max_chars: int) -> List[str]:
    if repo_to_chunks is not None:
        chs, _ = repo_to_chunks({title: text}, max_chars=max_chars)
        return chs or []
    chs, _ = fallback_to_chunks({title: text}, max_chars=max_chars)
    return chs

def load_hotpotqa_pages(path: str, num_examples: Optional[int] = None) -> Tuple[Dict[str, str], int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pages: Dict[str, str] = {}
    used = 0
    for ex in data:
        if num_examples is not None and used >= num_examples:
            break
        ctx = ex.get("context", [])
        if isinstance(ctx, list):
            for item in ctx:
                if not (isinstance(item, list) or isinstance(item, tuple)) or len(item) < 2:
                    continue
                title = str(item[0])
                sents = item[1]
                if title in pages:
                    continue
                if isinstance(sents, list):
                    pages[title] = " ".join([str(s).strip() for s in sents if str(s).strip()])
                else:
                    pages[title] = str(sents)
        used += 1
    return pages, used

def load_musique_pages(path: str, num_examples: Optional[int] = None) -> Tuple[Dict[str, str], int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pages: Dict[str, List[str]] = {}
    used = 0
    for ex in data:
        if num_examples is not None and used >= num_examples:
            break
        paras = ex.get("paragraphs", [])
        if isinstance(paras, list):
            for p in paras:
                if not isinstance(p, dict):
                    continue
                title = str(p.get("title", "")).strip()
                txt = str(p.get("paragraph_text", "")).strip()
                if not title or not txt:
                    continue
                if title not in pages:
                    pages[title] = [txt]
                else:
                    if txt not in pages[title]:
                        pages[title].append(txt)
        used += 1
    joined: Dict[str, str] = {t: "\n\n".join(v) for t, v in pages.items()}
    return joined, used

def load_pages_auto(path: str, num_examples: Optional[int] = None) -> Tuple[Dict[str, str], int, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Unsupported data format (expected non-empty list): {path}")

    first = data[0]
    if isinstance(first, dict) and "context" in first:
        pages, used = load_hotpotqa_pages(path, num_examples=num_examples)
        return pages, used, "hotpotqa"
    if isinstance(first, dict) and "paragraphs" in first:
        pages, used = load_musique_pages(path, num_examples=num_examples)
        return pages, used, "musique"

    raise ValueError(f"Unsupported data format (cannot detect): {path}")

def build_openie_kg_jsonl_incremental(
    pages: Dict[str, str],
    out_dir: str,
    num_pages: int = 2000,
    topk_chunks: int = 4,
    add_community: bool = True,
    community_algo: str = "label_prop",
    chunk_chars: int = 1200,
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
    ent_key2id: Dict[str, int] = {}
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
                max_eid = max(max_eid, eid)
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
                triple_seen.add((h_id, t_id, normalize(rel), title))
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

    def get_or_create_entity(name: str, etype: str = "OTHER") -> Optional[int]:
        nonlocal entities_count
        key = normalize(name)
        if not key:
            return None
        if key in ent_key2id:
            return ent_key2id[key]
        eid = entities_count
        entities_count += 1
        append_jsonl(entities_path, {"id": eid, "name": name, "type": normalize_type(etype)})
        ent_key2id[key] = eid
        return eid

    for idx in tqdm(range(start_title_idx, len(titles)), desc="Building OpenIE-fast KG (pages)"):
        title = titles[idx]
        if title in processed_titles:
            continue

        text = pages.get(title, "")
        chunks = make_chunks(title, text, max_chars=chunk_chars)
        if not chunks:
            if (idx + 1) % checkpoint_every_titles == 0:
                atomic_write_json(ckpt_path, {
                    "status": "running",
                    "next_title_idx": idx + 1,
                    "entities_count": entities_count,
                    "triples_count": triples_count
                })
            continue

        per_chunk: List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]] = []
        for c in chunks:
            ents, rels = llm_openie_extract(llm, c)
            per_chunk.append((ents, rels))

        title_entities_set = set()
        title_triples_list: List[int] = []

        ents_local: Dict[str, Dict[str, str]] = {}
        for ents, _ in per_chunk:
            for e in ents:
                k = normalize(e.get("name", ""))
                if k and k not in ents_local:
                    ents_local[k] = {"name": e["name"], "type": normalize_type(e.get("type", "OTHER"))}

        for e in ents_local.values():
            eid = get_or_create_entity(e["name"], e.get("type", "OTHER"))
            if eid is not None:
                title_entities_set.add(eid)

        for _, rels in per_chunk:
            for r in rels:
                h_name = (r.get("head") or "").strip()
                t_name = (r.get("tail") or "").strip()
                rel = (r.get("relation") or "").strip()
                if not (h_name and t_name and rel):
                    continue

                h_id = get_or_create_entity(h_name)
                t_id = get_or_create_entity(t_name)
                if h_id is None or t_id is None:
                    continue

                key_tr = (h_id, t_id, normalize(rel), title)
                if key_tr in triple_seen:
                    continue
                triple_seen.add(key_tr)

                triple_idx = triples_count
                triples_count += 1

                append_jsonl(triples_path, {
                    "h": h_id,
                    "t": t_id,
                    "r": rel,
                    "evidence": (r.get("evidence") or "").strip(),
                    "title": title
                })

                title_triples_list.append(triple_idx)
                title_entities_set.add(h_id)
                title_entities_set.add(t_id)

        append_jsonl(t2e_path, {"title": title, "entity_ids": sorted(list(title_entities_set))})
        append_jsonl(t2t_path, {"title": title, "triple_idxs": title_triples_list})
        processed_titles.add(title)

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
        "out_dir": str(out_dir),
        "chunk_chars": chunk_chars
    })

    print(f"[openie_builder_fast] Saved to {out_dir}")
    print(f"[openie_builder_fast] entities={entities_count}, triples={triples_count}")
    print("[openie_builder_fast] files:")
    print(f"  - {entities_path}")
    print(f"  - {triples_path}")
    print(f"  - {t2e_path}")
    print(f"  - {t2t_path}")
    print(f"  - {ckpt_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
         nargs="+",
        default=['datasets/dataset/musique.json'], 
        help="One or more dataset json files (e.g., hotpotqa.json musique.json). Auto-detect format."
    )
    ap.add_argument(
        "--num_examples",
        type=int,
        default=1000,
        help="Use only first N examples from EACH input file to build the page corpus."
    )
    ap.add_argument(
        "--out",
        default="",
        help="Output directory for JSONL files."
    )
    ap.add_argument(
        "--chunk_chars",
        type=int,
        default=600,
        help="Max characters per chunk (larger => fewer LLM calls)."
    )
    ap.add_argument("--checkpoint_every_titles", type=int, default=10)
    ap.add_argument("--no_resume", action="store_true")

    ap.add_argument("--num_pages", type=int, default=None, help="(optional) limit #titles processed")
    ap.add_argument("--topk_chunks", type=int, default=4, help="(kept for compatibility; ignored)")
    ap.add_argument("--no_community", action="store_true", help="(kept for compatibility; ignored)")

    args = ap.parse_args()

    pages_all: Dict[str, str] = {}
    fmt_stats: List[Tuple[str, int, int, str]] = []

    for p in args.inputs:
        pages, used, fmt = load_pages_auto(p, num_examples=args.num_examples)
        new_cnt = 0
        for title, text in pages.items():
            if title not in pages_all:
                pages_all[title] = text
                new_cnt += 1
        fmt_stats.append((p, used, new_cnt, fmt))

    print("[openie_builder_fast] Collected pages:")
    for p, used, new_cnt, fmt in fmt_stats:
        print(f"  - {p} ({fmt}): used_examples={used}, new_pages={new_cnt}")
    print(f"  => total_unique_pages={len(pages_all)}")

    n_titles = args.num_pages if args.num_pages is not None else len(pages_all)
    build_openie_kg_jsonl_incremental(
        pages_all,
        out_dir=args.out,
        num_pages=n_titles,
        topk_chunks=args.topk_chunks,
        add_community=(not args.no_community),
        community_algo="label_prop",
        chunk_chars=args.chunk_chars,
        checkpoint_every_titles=args.checkpoint_every_titles,
        resume=(not args.no_resume),
    )

if __name__ == "__main__":
    main()