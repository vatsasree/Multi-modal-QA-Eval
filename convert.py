import json

# ---- Paths ----
old_json_path = '/projects/data/vision-team/shanmukha_sreevatsa/Patram/qa_pairs_with_estimated_times_patramparse_oss.json'
new_json_path = '/projects/data/vision-team/shanmukha_sreevatsa/Patram/qa_pairs_v2_grouped.json'
output_path   = '/projects/data/vision-team/shanmukha_sreevatsa/Patram/qa_pairs_v2_grouped_timed.json'

# ---- Load JSONs ----
with open(old_json_path, 'r') as f:
    old_data = json.load(f)

with open(new_json_path, 'r') as f:
    new_data = json.load(f)

# ---- Build lookup dict from old JSON ----
# mapping: (img_path, question) -> timing fields
lookup = {}
for doc in old_data:
    img_path = doc["img_path"]
    for qa in doc["qa_pairs"]:
        q = qa["question"].strip()
        lookup[(img_path, q)] = {
            "model1_time": qa.get("model1_time"),
            "model2_time_patram_parse": qa.get("model2_time_patram_parse"),
            "model2_oss_time": qa.get("model2_oss_time")
        }

# ---- Merge into new JSON ----
for img_path, doc in new_data.items():
    for qa in doc["qa_pairs"]:
        q = qa["question"].strip()
        key = (img_path, q)
        if key in lookup:
            qa.update(lookup[key])

# ---- Save merged JSON ----
with open(output_path, 'w') as f:
    json.dump(new_data, f, indent=4)

print(f"Merged JSON saved to {output_path}")


#####################################################################################################################

# adding num_crops

import json
import glob

# ---- Load existing JSON ----
json_path = '/projects/data/vision-team/shanmukha_sreevatsa/Patram/qa_pairs_v2_grouped_timed.json'
with open(json_path, 'r') as f:
    json_data = json.load(f)

# ---- Add num_crops to each document ----
for img_path, doc in json_data.items():
    doc_id = doc["doc_id"]
    doc_folder = img_path.split('/')[-1].split('.')[0]
    
    # Count crops
    template_folder = f'/projects/data/vision-team/aryan_jain/BGenV-Data_viewer/IMC_data/{doc_folder}/crops/*_viz.png'
    num_crops = len(glob.glob(template_folder))
    
    # Add key
    doc["num_crops"] = num_crops

# ---- Save updated JSON ----
output_path = '/projects/data/vision-team/shanmukha_sreevatsa/Patram/qa_pairs_v2_grouped_timed.json'
with open(output_path, 'w') as f:
    json.dump(json_data, f, indent=4)

print(f"Updated JSON with num_crops saved to {output_path}")