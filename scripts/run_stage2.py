
from ..configs.config import parse_args
from ..configs.io import load_json, save_json
from ..pipelines.pipeline2 import run_stage2

def main():
    args = parse_args()

    stage1_items = load_json(args.stage1_json)
    if not isinstance(stage1_items, list):
        raise ValueError("stage1_json must be a JSON list.")

    if args.num_samples is not None:
        stage1_items = stage1_items[: int(args.num_samples)]

    results = run_stage2(args.dataset, stage1_items, args.data_json, args)
    save_json(args.stage2_json, results, indent=2)
    print("Saved:", args.stage2_json)

if __name__ == "__main__":
    main()
