"""Combined UOIS score = 0.4 * Objects F + 0.4 * Boundary F + 0.2 * (fraction of objects detected at 0.75).

Usage: python combined_score.py <results.json> [<results.json> ...]
Accepts the flat results.json written by iteach_test_dataset.py (HumanPlay test set)
and the nested {"<dataset>": {"results": {...}}} format.
"""
import os
import json
import sys
from collections import defaultdict

# Datasets of interest
datasets = ['ocid', 'osd', 'pushing', 'iteach-uois']

# Weighted combination function
def compute_combined_score(obj_f, bnd_f, det_075):
    return 0.4 * obj_f + 0.4 * bnd_f + 0.2 * det_075

# All results: {dataset: [(model_name, score), ...]}
scores_by_dataset = defaultdict(list)

# Loop through all input JSON files
for json_file in sys.argv[1:]:
    with open(json_file, 'r') as f:
        data = json.load(f)

    model_name = data.get("model", os.path.splitext(os.path.basename(json_file))[0])
    # iteach_test_dataset.py / iteach_get_results_all_models.py write a flat dict of
    # metrics for the HumanPlay test set; treat that as the 'iteach-uois' entry.
    if "Objects F-measure" in data:
        data = {"iteach-uois": {"results": data}}
        if model_name == "results":  # MSMFormer/<out_dir>/model_results/results.json
            model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(json_file))))
    first = next((v for v in data.values() if isinstance(v, dict)), {})
    results_key = "results_refined" if "results_refined" in first else "results"

    for dataset in datasets:
        if dataset not in data or results_key not in data[dataset]:
            continue

        results = data[dataset][results_key]
        obj_f = results.get("Objects F-measure", 0)
        bnd_f = results.get("Boundary F-measure", 0)
        det_075 = results.get("obj_detected_075_percentage", 0)

        combined = compute_combined_score(obj_f, bnd_f, det_075)
        scores_by_dataset[dataset].append((model_name, combined))

# Display top 3 per dataset
print("\nTop 3 Models per Dataset (based on combined score):\n" + "="*50)
for dataset in datasets:
    print(f"\n{dataset.upper()}:")
    top_models = sorted(scores_by_dataset[dataset], key=lambda x: x[1], reverse=True)[:5]
    for rank, (name, score) in enumerate(top_models, 1):
        # metrics are fractions in [0, 1]; x100 matches the tables made by j2trex.py
        print(f"  {rank}. {name} – {score:.4f} ({100 * score:.1f})")
