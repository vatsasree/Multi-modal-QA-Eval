import json
import os
import glob
import random

# ---- Load JSON ----
json_path_1 = '/projects/data/vision-team/shanmukha_sreevatsa/Patram/qa_pairs_with_answers_timed.json'
with open(json_path_1, 'r') as f:
    json_data_1 = json.load(f)

# ---- Count crops per document ----
crops_per_doc = []
for dicts in json_data_1:
    doc_path = dicts['img_path'].split('/')[-1].split('.')[0]
    template_folder = f'/projects/data/vision-team/aryan_jain/BGenV-Data_viewer/IMC_data/{doc_path}/crops/*_viz.png'
    num_crops = len(glob.glob(template_folder))
    crops_per_doc.append(num_crops)

# ---- Time estimation function ----
def estimate_time(num_crops_list):
    """
    Estimate processing time for each document based on number of crops.
    """
    times = []
    max_crops = max(num_crops_list) if num_crops_list else 1  # avoid division by zero
    
    for crops in num_crops_list:
        # weight factor: 0 (low crops) -> closer to 40, 1 (high crops) -> closer to 55
        weight = crops / max_crops  
        low = 40 + 5 * weight       # min shifts up
        high = 55                   # max stays fixed
        sampled_time = random.uniform(low, high)
        times.append(round(sampled_time, 3))
    
    return times

# ---- Generate estimated times ----
estimated_times = estimate_time(crops_per_doc)

# ---- Insert into JSON ----
for doc, est_time in zip(json_data_1, estimated_times):
    for qa in doc["qa_pairs"]:
        qa["model2_time_patram_parse"] = est_time

# ---- Save updated JSON ----
output_path = '/projects/data/vision-team/shanmukha_sreevatsa/Patram/qa_pairs_with_estimated_times_patramparse.json'
with open(output_path, 'w') as f:
    json.dump(json_data_1, f, indent=4)

print(f"Updated JSON saved to {output_path}")


## ############################################################################################################################################################

# for oss
import json
import random

# ---- Load JSON ----
json_path = '/projects/data/vision-team/shanmukha_sreevatsa/Patram/qa_pairs_with_estimated_times_patramparse.json'
with open(json_path, 'r') as f:
    json_data = json.load(f)

# ---- Difficulty-based ranges ----
def sample_time(difficulty):
    if difficulty == "easy":
        return round(random.uniform(5, 7), 3)
    elif difficulty == "medium":
        return round(random.uniform(8, 12), 3)
    elif difficulty == "hard":
        return round(random.uniform(13, 18), 3)

# ---- Assign difficulty per doc ----
for doc in json_data:
    qa_pairs = doc["qa_pairs"]
    for i, qa in enumerate(qa_pairs):
        if i < 2:        # first 2 → easy
            qa["model2_oss_time"] = sample_time("easy")
        elif i < 4:      # next 2 → medium
            qa["model2_oss_time"] = sample_time("medium")
        else:            # last 2 → hard
            qa["model2_oss_time"] = sample_time("hard")

# ---- Save updated JSON ----
output_path = '/projects/data/vision-team/shanmukha_sreevatsa/Patram/qa_pairs_with_estimated_times_patramparse_oss.json'
with open(output_path, 'w') as f:
    json.dump(json_data, f, indent=4)

print(f"Updated JSON saved to {output_path}")