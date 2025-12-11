import json
import glob
from transformers import pipeline

# ----------------
# Load model
# ----------------
model_id = "openai/gpt-oss-20b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="bfloat16",
    device_map="auto"
    # device_map = ''
)

def get_oss_response(pipe, html_str, question):
#     prompt = f"""Given a HTML {html_str}. Answer the following question. 
# Question: {question}
# Provide the final answer only in <ANSWER>.
# """
    prompt = f"""
        You are an expert at reading and understanding structured HTML content.

        Your task is to answer the question **based only on the provided HTML content**.

        HTML Content:
        {html_str}

        Question:
        {question}

        Instructions:
        - Carefully read the HTML content.
        - Extract the exact factual answer relevant to the question.
        - Do **not** include explanations, reasoning, or restate the question.
        - If the answer is not found, output: <ANSWER>Not found</ANSWER>.
        - Always return the final answer **only once**, wrapped strictly in XML tags:
        <ANSWER>...</ANSWER>
    """
    outputs = pipe(
        prompt,
        max_new_tokens=1500,
        return_full_text=False,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    summary = outputs[0]["generated_text"].strip()
    return summary

# ----------------
# Paths
# ----------------
html_folder = '/projects/data/vision-team/aryan_jain/BGenV-Data_viewer/IMC_data'
json_path = '/projects/data/vision-team/shanmukha_sreevatsa/Patram/qa_pairs_with_answers.json'
output_json_path = '/projects/data/vision-team/shanmukha_sreevatsa/Patram/qa_pairs_with_answers_oss_prompt_2.json'

# ----------------
# Load JSON
# ----------------
json_data = json.load(open(json_path, 'r'))

# ----------------
# Loop over QA entries
# ----------------
from tqdm import tqdm
import glob
import json

# Assuming json_data, html_folder, get_oss_response, and pipe are already defined

for qa_dict in tqdm(json_data, desc="Processing JSON entries"):
    path = qa_dict['img_path']
    folder = path.split('/')[-1].split('.')[0]

    html_files = glob.glob(f'{html_folder}/{folder}/reading_order/*_caption.html')
    if not html_files:
        print(f"No HTML found for {folder}")
        continue

    html_path = html_files[0]  # pick first match
    with open(html_path, 'r') as f:
        html_str = f.read()

    qa_pairs = qa_dict['qa_pairs']
    for qa_pair in tqdm(qa_pairs, desc=f"QA pairs for {folder}", leave=False):
        question = qa_pair['question']
        answer = get_oss_response(pipe, html_str, question)
        qa_pair['model2'] = answer


# ----------------
# Save output
# ----------------
with open(output_json_path, 'w') as f:
    json.dump(json_data, f, indent=2)

print("Saved results to:", output_json_path)
