import json
import os
import subprocess



dir_path = os.path.dirname(os.path.realpath(__file__))
metadata_path = "metadata.json"
metadata_path = os.path.join(dir_path, metadata_path)

script_path = os.path.join(dir_path, "evaluate_trigger_candidates.py")

with open(metadata_path, 'r') as file:
    data = json.load(file)

# Accessing the data
for model_key, model_info in data.items():
    print(f"Model: {model_key}")
    print(f"  Ground Truth Trigger: {model_info['ground_truth_trigger']}")
    print(f"  Triggers: {model_info['triggers']}")
    print(f"  Rewards: {model_info['rewards']}")
    print(f"  Model Name: {model_info['model_name']}")

    for t in model_info['triggers']:
        subprocess.run(['python', script_path, '--generation_model_name', model_info['model_name'], '--trigger', t])
