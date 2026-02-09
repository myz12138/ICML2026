
import os
import argparse

EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMB_DEVICE = os.getenv("EMB_DEVICE", "cpu")

TOP_N = int(os.getenv("TOP_N_LLM", "3"))
TOP_K_ENTITY = int(os.getenv("TOP_K_ENTITY", "3"))
RELATION_K = int(os.getenv("RELATION_K", "8"))
NEFF_THRESHOLD = float(os.getenv("NEFF_THRESHOLD", "1.5"))

RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "128"))
RERANK_BATCH = int(os.getenv("RERANK_BATCH", "32"))
RERANK_FP16 = os.getenv("RERANK_FP16", "1") == "1"
INCLUDE_QUESTION_IN_RERANK = os.getenv("INCLUDE_QUESTION_IN_RERANK", "1") == "1"

TOPK_RESOLVED = int(os.getenv("TOPK_RESOLVED", "3"))
TOPK_UNRESOLVED = int(os.getenv("TOPK_UNRESOLVED", "3"))
GLOBAL_TOPK_RESOLVED = int(os.getenv("GLOBAL_TOPK_RESOLVED", "20"))
GLOBAL_TOPK_UNRESOLVED = int(os.getenv("GLOBAL_TOPK_UNRESOLVED", "20"))
DOC_PREFILTER_K = int(os.getenv("DOC_PREFILTER_K", "80"))
UNRES_TOP_SIMILAR = int(os.getenv("UNRES_TOP_SIMILAR", "20"))

_DEFAULTS = {
    "2wiki": {
        "QUERY_JSON": "result_1.5/2wiki_data_100/query_graph_v8_2wiki.json",
        "STAGE1_JSON": "result_1.5/2wiki_data_100/stage1_struct_v8_2wiki.json",
        "DATA_JSON": "dataset/2wikimultihopqa.json",
        "STAGE2_JSON": "result_1.5/2wiki_data_100/stage2_evidence_v8_2wiki.json",
        "KG_PATH": "seedkg_2wiki_1000_jsonl_2step",
    },
    "hotpotqa": {
        "QUERY_JSON": "result_1.5/hotpotqa_data_100/query_graph_v8_hotpotqa.json",
        "STAGE1_JSON": "result_1.5/hotpotqa_data_100/stage1_struct_v8_hotpotqa.json",
        "DATA_JSON": "dataset/hotpotqa.json",
        "STAGE2_JSON": "result_1.5/hotpotqa_data_100/stage2_evidence_v8_hotpotqa.json",
        "KG_PATH": "seedkg_hotpotqa_1000_jsonl_2step_new",
    },
    "musique": {
        "QUERY_JSON": "result_1.5/musique_data_100/query_graph_v8_musique.json",
        "STAGE1_JSON": "result_1.5/musique_data_100/stage1_struct_v8_musique.json",
        "DATA_JSON": "dataset/musique.json",
        "STAGE2_JSON": "result_1.5/musique_data_100/stage2_evidence_v8_musique.json",
        "KG_PATH": "seedkg_musique_1000_jsonl",
    },
}

def _none_or_int(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("none", "null"):
        return None
    return int(x)

def build_parser():
    parser = argparse.ArgumentParser(description="C2RAG Two-Stage Retrieval")

    parser.add_argument("--dataset", type=str, choices=sorted(list(_DEFAULTS.keys())), default="2wiki")
    parser.add_argument("--num_samples", type=_none_or_int, default=1000)

    parser.add_argument("--query_json", type=str, default=_DEFAULTS["2wiki"]["QUERY_JSON"])
    parser.add_argument("--stage1_json", type=str, default=_DEFAULTS["2wiki"]["STAGE1_JSON"])
    parser.add_argument("--data_json", type=str, default=_DEFAULTS["2wiki"]["DATA_JSON"])
    parser.add_argument("--stage2_json", type=str, default=_DEFAULTS["2wiki"]["STAGE2_JSON"])
    parser.add_argument("--kg_path", type=str, default=_DEFAULTS["2wiki"]["KG_PATH"])

    parser.add_argument("--emb_model", type=str, default=EMB_MODEL)
    parser.add_argument("--emb_device", type=str, default=EMB_DEVICE)

    parser.add_argument("--top_n", type=int, default=TOP_N)
    parser.add_argument("--top_k_entity", type=int, default=TOP_K_ENTITY)
    parser.add_argument("--relation_k", type=int, default=RELATION_K)
    parser.add_argument("--neff_threshold", type=float, default=NEFF_THRESHOLD)

    parser.add_argument("--rerank_model", type=str, default=RERANK_MODEL)
    parser.add_argument("--rerank_max_length", type=int, default=RERANK_MAX_LENGTH)
    parser.add_argument("--rerank_batch", type=int, default=RERANK_BATCH)
    parser.add_argument("--rerank_fp16", type=int, default=1 if RERANK_FP16 else 0)
    parser.add_argument("--include_question_in_rerank", type=int, default=1 if INCLUDE_QUESTION_IN_RERANK else 0)

    parser.add_argument("--topk_resolved", type=int, default=TOPK_RESOLVED)
    parser.add_argument("--topk_unresolved", type=int, default=TOPK_UNRESOLVED)
    parser.add_argument("--global_topk_resolved", type=int, default=GLOBAL_TOPK_RESOLVED)
    parser.add_argument("--global_topk_unresolved", type=int, default=GLOBAL_TOPK_UNRESOLVED)
    parser.add_argument("--doc_prefilter_k", type=int, default=DOC_PREFILTER_K)
    parser.add_argument("--unres_top_similar", type=int, default=UNRES_TOP_SIMILAR)

    return parser

def apply_dataset_defaults(args):
    d = _DEFAULTS.get(args.dataset, {})
    if args.query_json == _DEFAULTS["2wiki"]["QUERY_JSON"]:
        args.query_json = d.get("QUERY_JSON", args.query_json)
    if args.stage1_json == _DEFAULTS["2wiki"]["STAGE1_JSON"]:
        args.stage1_json = d.get("STAGE1_JSON", args.stage1_json)
    if args.data_json == _DEFAULTS["2wiki"]["DATA_JSON"]:
        args.data_json = d.get("DATA_JSON", args.data_json)
    if args.stage2_json == _DEFAULTS["2wiki"]["STAGE2_JSON"]:
        args.stage2_json = d.get("STAGE2_JSON", args.stage2_json)
    if args.kg_path == _DEFAULTS["2wiki"]["KG_PATH"]:
        args.kg_path = d.get("KG_PATH", args.kg_path)
    return args

def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    return apply_dataset_defaults(args)
