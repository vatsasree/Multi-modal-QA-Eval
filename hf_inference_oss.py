from transformers.distributed import DistributedConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import glob
import json
 
def split_batch(lst, n):
    """
    Split a list `lst` into `n` roughly equal parts.
    Returns a list of lists.
    """
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

# model_path = "openai/gpt-oss-20b"
# tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
 
# device_map = {
#     # Enable Expert Parallelism
#     "distributed_config": DistributedConfig(enable_expert_parallel=1),
#     # Enable Tensor Parallelism
#     "tp_plan": "auto",
# }
 
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype="auto",
#     attn_implementation="kernels-community/vllm-flash-attn3",
#     **device_map,
# )

model_name = "openai/gpt-oss-20b"
 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda:5"
)
 

def get_oss_response(model, tokenizer, html_str, question):
     
    prompt = f"""
        You are an expert at reading and understanding structured HTML content.

        Your task is to answer the question **based only on the provided HTML content**.

        HTML Content:
        {html_str}

        Question:
        {question}

        Instructions:
        - Carefully read the HTML content.
        - Extract the exact factual answer relevant to the question using all the information from the HTML. 
    """

    messages = [
        {"role": "system", "content": "You are expert at understanding HTML and CSS. "},
        {"role": "user", "content": prompt}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=1000) 
    # Decode and print
    response = tokenizer.decode(outputs[0])
    print("Model response:", response.split("<|channel|>final<|message|>")[-1].strip())

    return response.split("<|channel|>final<|message|>")[-1].strip()

# ----------------
# Paths
# ----------------
html_folder = '/projects/data/vision-team/aryan_jain/BGenV-Data_viewer/IMC_data'
json_path = '/projects/data/vision-team/shanmukha_sreevatsa/Patram/qa_pairs_with_answers.json'
output_json_path = '/projects/data/vision-team/shanmukha_sreevatsa/Patram/qa_pairs_with_answers_oss_v2_1.json'

# ----------------
# Load JSON
# ----------------
json_data_full = json.load(open(json_path, 'r'))
json_data = split_batch(json_data_full, 2)[1]

# ----------------
# Loop over QA entries
# ----------------


# Assuming json_data, html_folder, get_oss_response, and pipe are already defined

for qa_dict in tqdm(json_data, desc="Processing JSON entries"):
    path = qa_dict['img_path']
    folder = path.split('/')[-1].split('.')[0]

    html_files = glob.glob(f'{html_folder}/{folder}/reading_order/*_caption.html')
    if not html_files:
        print(f"No HTML found for {folder}")
        continue

    # html_path = html_files[0]  # pick first match
    for html_path in html_files:
        with open(html_path, 'r') as f:
            html_str = f.read()

        qa_pairs = qa_dict['qa_pairs']
        for qa_pair in tqdm(qa_pairs, desc=f"QA pairs for {folder}", leave=False):
            question = qa_pair['question']
            answer = get_oss_response(model, tokenizer, html_str, question)
            # import pdb; pdb.set_trace()
            qa_pair['model2'] = answer


# ----------------
# Save output
# ----------------
with open(output_json_path, 'w') as f:
    json.dump(json_data, f, indent=2)

print("Saved results to:", output_json_path)
