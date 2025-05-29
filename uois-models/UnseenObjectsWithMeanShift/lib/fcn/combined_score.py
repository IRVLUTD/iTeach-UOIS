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
    results_key = "results_refined" if "results_refined" in next(iter(data.values()), {}) else "results"

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
        print(f"  {rank}. {name} – {score:.2f}")
