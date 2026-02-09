import os
import json
import glob
import torch

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_jsonl_files(paths):
    items = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
    return items

def _fallback_load_kg_from_jsonl_dir(dir_path):
    # Best-effort fallback: expects jsonl lines to contain either 'entities'/'triples'
    # or KG triple dicts with integer ids. If your repo has read_hotpotqa.load_kg_from_jsonl,
    # that will be used instead.
    jsonl_files = sorted(glob.glob(os.path.join(dir_path, "*.jsonl")))
    if not jsonl_files:
        raise RuntimeError(f"No .jsonl files found under {dir_path}")
    items = _load_jsonl_files(jsonl_files)

    # If a single dict contains entities/triples, return it.
    for it in items:
        if isinstance(it, dict) and "entities" in it and "triples" in it:
            return it

    # Otherwise, attempt to build a trivial KG with only triple dicts and entity name strings.
    ent2id = {}
    entities = []
    triples = []
    def _get_eid(name):
        name = str(name)
        if name not in ent2id:
            ent2id[name] = len(entities)
            entities.append({"name": name})
        return ent2id[name]

    for it in items:
        if not isinstance(it, dict):
            continue
        h = it.get("head") or it.get("h") or it.get("subject")
        r = it.get("relation") or it.get("r") or it.get("predicate")
        t = it.get("tail") or it.get("t") or it.get("object")
        if h is None or r is None or t is None:
            continue
        triples.append({"head": _get_eid(h), "relation": str(r), "tail": _get_eid(t), **{k:v for k,v in it.items() if k not in ("head","h","subject","relation","r","predicate","tail","t","object")}})
    return {"entities": entities, "triples": triples}

def load_kg(kg_path):
    if isinstance(kg_path, str) and kg_path.endswith(".pt"):
        return torch.load(kg_path)

    if isinstance(kg_path, str) and (kg_path.endswith(".json") or kg_path.endswith(".jsonl")) and os.path.isfile(kg_path):
        kg = _load_json(kg_path)
        if isinstance(kg, dict) and "entities" in kg and "triples" in kg:
            return kg

    # Prefer your existing loader if present
    try:
        from read_hotpotqa import load_kg_from_jsonl
        return load_kg_from_jsonl(kg_path)
    except Exception:
        pass

    # Fallback: if kg_path is a directory of jsonl files
    if isinstance(kg_path, str) and os.path.isdir(kg_path):
        return _fallback_load_kg_from_jsonl_dir(kg_path)

    raise RuntimeError(f"Failed to load KG from {kg_path}")
