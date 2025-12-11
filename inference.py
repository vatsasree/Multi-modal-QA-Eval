# working script to get answers from Patram-7B and save to structured json

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
from PIL import Image
import json
import time

# Model ID and device setup
model_id = '/projects/data/vision-team/venkat_kesav/GR_Model_Training_with_Swift/final_outputs/batch_7-14_200k_docs/output_5555/v0-20250708-204650/checkpoint-22004'
device = "cuda:2" if torch.cuda.is_available() else "cpu"

# Load processor and model
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True
).to(device)

def get_patram_response(image_path, question):
    try:
        # Load local image
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # Format the prompt
    prompt = f"Question: {question} Answer based on the image."

    try:
        # Preprocess image + text
        inputs = processor.process(images=[image], text=prompt)
        inputs = {k: v.to(device).unsqueeze(0) for k, v in inputs.items()}

        # Generate output
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

        # Decode response
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        response = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return response
    except Exception as e:
        print(f"Error during inference: {e}")
        return None


# Example usage with a local image
# image_input = "/projects/data/vision-team/shanmukha_sreevatsa/IMC/finance_7.png"
# question = "Who issued this notice?"
# answer = get_patram_response(image_input, question)
# if answer:
#     print("Answer:", answer)


qa_json_path = '/projects/data/vision-team/shanmukha_sreevatsa/Patram/qa_pairs.json'
qa_data = json.load(open(qa_json_path,'r'))
import tqdm

for doc in tqdm.tqdm(qa_data, desc="Documents"):
    img_path = doc['img_path']
    for i, qa in tqdm.tqdm(enumerate(doc["qa_pairs"]), total=len(doc["qa_pairs"]), desc=f"Questions in {doc['doc_id']}", leave=False):
        question = qa["question"]

        start_time = time.time()
        answer = get_patram_response(img_path, question)
        end_time = time.time()

        qa["model1"] = answer if answer else ""
        qa["model1_time"] = round(end_time - start_time, 3)  # seconds with 3 decimals



# Save back the updated JSON
out_path = "/projects/data/vision-team/shanmukha_sreevatsa/Patram/qa_pairs_with_answers_timed.json"
with open(out_path, "w") as f:
    json.dump(qa_data, f, indent=4)