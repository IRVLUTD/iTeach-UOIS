import json
import sys
import os

# Get JSON file from command line
json_file = sys.argv[1]

# Load JSON data
with open(json_file, 'r') as f:
    data = json.load(f)

# Datasets and metric keys
# datasets = ['ocid', 'osd', 'pushing', 'iteach-uois']
metric_keys = [
    ('Objects Precision', 'Objects Recall', 'Objects F-measure'),
    ('Boundary Precision', 'Boundary Recall', 'Boundary F-measure')
]

datasets = ['iteach-uois']
# metric_keys = [
#     ('Objects F-measure','Boundary F-measure')
# ]


# Format number to one decimal
fmt = lambda x: f"{x * 100:.1f}"

# Get model name from JSON or fallback to filename
base_model_name = data.get("model", os.path.splitext(os.path.basename(json_file))[0])

# Function to capture the 75% value and other metrics
def get_additional_values(results):
    # Get the obj_detected_075_percentage and return it
    return fmt(results.get("obj_detected_075_percentage", 0))

# Generate row from results dict
def make_row(results_key: str, label_suffix: str):
    row = r'\multicolumn{1}{c|}{' + base_model_name + label_suffix + '}'
    final_dataset_idx = len(datasets) - 1
    for idx, dataset in enumerate(datasets):
        if dataset not in data or results_key not in data[dataset]:
            row += " & " + " & ".join(["-"] * 7) + " & -"  # Add 7 + extra column for 75% and border
            continue

        results = data[dataset][results_key]
        
        for metric_group in metric_keys:
            for i, key in enumerate(metric_group):
                print(key)
                val = fmt(results.get(key, 0))
                if i == 2:
                    row += f" & \\multicolumn{{1}}{{c|}}{{{val}}}"
                else:
                    row += f" & {val}"

        # Add the 75% value without a vertical border
        additional_val = get_additional_values(results)
        if idx == final_dataset_idx:
            row += f" & {additional_val} "  # No multicolumn or border here
        else:
            row += f" & \\multicolumn{{1}}{{c|}}{{{additional_val}}} "  # No multicolumn or border here
        
    # End the row
    row += r" \\ \cline{1-1}"

    return row

# Print both rows
print()
print()
print("1st-Stage")
print()
print(make_row("results", "-raw"))
print()
print()
# print()
# print("2nd-Stage")
# print()
# print(make_row("results_refined", "-refined"))
# print()
