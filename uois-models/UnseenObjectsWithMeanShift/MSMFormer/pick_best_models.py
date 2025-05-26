import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

def load_results(json_dir):
    results = []
    for file in os.listdir(json_dir):
        if file.endswith(".json"):
            file_path = os.path.join(json_dir, file)
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        for ckpt, values in data.items():
                            try:
                                if isinstance(values, dict) and "results_refined" in values:
                                    obj_fmeasure = values["results_refined"].get("Objects F-measure", 0)
                                    boundary_fmeasure = values["results_refined"].get("Boundary F-measure", 0)
                                    obj_75 = values["results_refined"].get("obj_detected_075_percentage", 0)
                                    combined_score = (0.4 * obj_fmeasure) + (0.4 * boundary_fmeasure) + (0.2 * obj_75)
                                    iter_num = int(ckpt.split('_')[1].split('.')[0])  # Extract iteration number
                                    results.append((ckpt, iter_num, combined_score, file_path))
                            except:
                                pass
                    else:
                        print(f"Skipping {file_path}: JSON is not a dictionary.")
                except json.JSONDecodeError as e:
                    print(f"Error loading {file_path}: {e}")
    return results

def find_top_models(json_dir, top_n=5):
    results = load_results(json_dir)
    results.sort(key=lambda x: x[2], reverse=True)  # Sort by combined score (descending)
    return results[:top_n]

def plot_scores(json_dir):
    results = load_results(json_dir)
    results.sort(key=lambda x: x[1])  # Sort by iteration number
    
    iterations = np.array([r[1] for r in results])
    scores = np.array([r[2] for r in results])
    
    plt.plot(iterations, scores, linestyle='-', marker='o', color='#1f77b4', alpha=0.8, label='Score Trend')
    
    # Compute and plot Mean Line
    mean_score = np.mean(scores)
    plt.axhline(y=mean_score, color='red', linestyle='--', label='Mean Score')
    
    plt.xlabel("Iteration")
    plt.ylabel("Combined Score")
    plt.title("Score vs Iteration")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <json_directory>")
        sys.exit(1)
    
    json_dir = sys.argv[1]  # Directory provided as a command-line argument
    if not os.path.isdir(json_dir):
        print(f"Error: {json_dir} is not a valid directory.")
        sys.exit(1)
    
    top_models = find_top_models(json_dir)
    
    print("Top 5 Best Models:")
    for rank, (ckpt, iter_num, score, path) in enumerate(top_models, 1):
        print(f"{rank}. {ckpt} (Iteration: {iter_num}, Score: {score:.6f}) - {path}")
    
    plot_scores(json_dir)
