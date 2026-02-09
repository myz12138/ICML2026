
import os
import argparse

_DEFAULTS = {
    "2wiki": {
        "QUERY_JSON": "result/2wiki/automic_query_2wiki.json",
        "STAGE1_JSON": "result/2wiki/stage1_2wiki.json",
        "DATA_JSON": "datasets/dataset/2wikimultihopqa.json",
        "STAGE2_JSON": "result/2wiki/stage2_2wiki.json",
        "KG_PATH": "datasets/kg_2wiki_1000_json",
    },
    "hotpotqa": {
        "QUERY_JSON": "result/hotpotqa/automic_query_hotpotqa.json",
        "STAGE1_JSON": "result/hotpotqa/stage1_hotpotqa.json",
        "DATA_JSON": "datasets/dataset/hotpotqa.json",
        "STAGE2_JSON": "result/hotpotqa/stage2_hotpotqa.json",
        "KG_PATH": "datasets/kg_hotpotqa_1000_json",
    },
    "musique": {
        "QUERY_JSON": "result/musique/automic_query_musique.json",
        "STAGE1_JSON": "result/musique/stage1_musique.json",
        "DATA_JSON": "datasets/dataset/musique.json",
        "STAGE2_JSON": "result/musique/stage2_musique.json",
        "KG_PATH": "datasets/kg_musique_1000_json",
    },
}

def build_parser():
    parser = argparse.ArgumentParser(description="C2RAG Two-Stage Retrieval")
    parser.add_argument("--api_key", type=str, default="sk-4ErcoxMACExg7eAe3bgpJyV4zXR7Znsis7X1KYjaucgMbElo")
    parser.add_argument("--base_url", type=str, default="https://api.holdai.top/v1/")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")

    parser.add_argument("--dataset", type=str, choices=sorted(list(_DEFAULTS.keys())), default="musique")
    args=parser.parse_args()  
    parser.add_argument("--num_samples", type=_none_or_int, default=1000, help="Set to None to process all.")
    parser.add_argument("--sleep_secs", type=float, default=0.5)

    parser.add_argument("--query_json", type=str, default=_DEFAULTS[args.dataset]["QUERY_JSON"])
    parser.add_argument("--stage1_json", type=str, default=_DEFAULTS[args.dataset]["STAGE1_JSON"])
    parser.add_argument("--data_json", type=str, default=_DEFAULTS[args.dataset]["DATA_JSON"])
    parser.add_argument("--stage2_json", type=str, default=_DEFAULTS[args.dataset]["STAGE2_JSON"])
    parser.add_argument("--kg_path", type=str, default=_DEFAULTS[args.dataset]["KG_PATH"])

    parser.add_argument("--emb_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--emb_device", type=str, default="cpu")

    parser.add_argument("--top_n", type=int, default="3")
    parser.add_argument("--top_k_entity", type=int, default="3")
    parser.add_argument("--relation_k", type=int, default="8")
    parser.add_argument("--neff_threshold", type=float, default="1.5")

    parser.add_argument("--rerank_model", type=str, default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--rerank_max_length", type=int, default="128")
    parser.add_argument("--rerank_batch", type=int, default="32")
    parser.add_argument("--rerank_fp16", type=int, default=1)
    parser.add_argument("--include_question_in_rerank", type=int, default=1)

    parser.add_argument("--topk_resolved", type=int, default="3")
    parser.add_argument("--topk_unresolved", type=int, default="3")
    parser.add_argument("--global_topk_resolved", type=int, default="20")
    parser.add_argument("--global_topk_unresolved", type=int, default="20")
    parser.add_argument("--doc_prefilter_k", type=int, default="80")
    parser.add_argument("--unres_top_similar", type=int, default="20")


    
    return parser


def _none_or_int(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("none", "null"):
        return None
    return int(x)

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

if __name__ == "__main__":
    parse_args()