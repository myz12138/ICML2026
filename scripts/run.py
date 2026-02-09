
from tqdm import tqdm

from ..configs.config import parse_args
from ..configs.io import load_json, load_raw_dataset_id2qa, normalize_query_items, save_json
from ..kg.loader import load_kg
from ..kg.index import build_kg_indices
from ..sim.embedding import build_embedder
from ..rerank.cross_encoder import build_reranker
from ..pipelines.pipeline1 import process_item_stage1
from ..pipelines.pipeline2 import run_stage2

def run_retrieval():
    args = parse_args()
    #load_data
    kg = load_kg(args.kg_path)
    entities, entity_names, triple_info, out_index, in_index = build_kg_indices(kg)

    raw_id2qa = load_raw_dataset_id2qa(args.dataset, args.data_json)
    query_items = load_json(args.query_json)
    if not isinstance(query_items, list):
        raise ValueError("query_json must be a JSON list.")

    if args.num_samples is not None:
        query_items = query_items[: int(args.num_samples)]
    #initilization
    query_items = normalize_query_items(args.dataset, query_items, id2qa=raw_id2qa)

    enc = build_embedder(args.emb_model, args.emb_device)
    reranker = build_reranker(args.rerank_model, fp16=bool(args.rerank_fp16))
    #processing
    results = []
    for item in tqdm(query_items, desc=f"Stage-1 ({args.dataset})"):
        results.append(process_item_stage1(item, kg, entities, entity_names, triple_info, out_index, in_index, enc, reranker, args))

    save_json(args.stage1_json, results, indent=2)
    print("Saved:", args.stage1_json)

    stage1_items = load_json(args.stage1_json)
    if not isinstance(stage1_items, list):
        raise ValueError("stage1_json must be a JSON list.")

    if args.num_samples is not None:
        stage1_items = stage1_items[: int(args.num_samples)]

    results = run_stage2(args.dataset, stage1_items, args.data_json, args)
    save_json(args.stage2_json, results, indent=2)
    print("Saved:", args.stage2_json)
if __name__ == "__main__":
    run_retrieval()
