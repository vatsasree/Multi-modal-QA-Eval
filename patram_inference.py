# import torch
# from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
# from PIL import Image

# # -------------------------
# # Device & dtype setup
# # -------------------------
# if torch.cuda.is_available():
#     device = "cuda:7"
#     torch_dtype = torch.float16  # FP16 for GPU
#     device_map = {"": 7}         # assign model to GPU 0 (change if multiple GPUs)
# else:
#     device = "cpu"
#     torch_dtype = torch.float32  # FP32 for CPU
#     device_map = {"": "cpu"}

# # -------------------------
# # Load processor and model
# # -------------------------
# # model_path = '/projects/data/vision-team/venkat_kesav/GR_Model_Training_with_Swift/final_outputs/batch_7-14_200k_docs/output_5555/v0-20250708-204650/checkpoint-22000'
# model_path = 'bharatgenai/patram-7b-instruct'

# processor = AutoProcessor.from_pretrained(
#     model_path,
#     trust_remote_code=True,
#     device_map=device_map
# )

# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     trust_remote_code=True,
#     device_map=device_map
# )

# # -------------------------
# # Function to get Patram response
# # -------------------------
# def get_patram_response(image_path, question, max_new_tokens=200):
#     """
#     Generate answer from Patram model for a local image and a text question.
    
#     Args:
#         image_path (str): Path to a local image file (png/jpg).
#         question (str): Question to ask about the image.
#         max_new_tokens (int): Maximum number of tokens to generate.
    
#     Returns:
#         str: Generated answer text or None if error occurs.
#     """
#     try:
#         # Load the local image
#         image = Image.open(image_path).convert("RGB")
#     except Exception as e:
#         print(f"Error loading image: {e}")
#         return None

#     # Format prompt
#     prompt = f"Question: {question} Answer based on the image."

#     # try:
#     # Preprocess image and text
#     import pdb; pdb.set_trace()
#     inputs = processor.process(images=[image], text=prompt)

#     # Ensure proper batch dimension and move to correct device
#     inputs = {
#         k: (v.unsqueeze(0).to(device) if v.ndim == 1 else v.to(device))
#         for k, v in inputs.items()
#     }
#     import pdb; pdb.set_trace()
#     # Generate output using Patram-specific method
#     output = model.generate_from_batch(
#         inputs,
#         generation_config=GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"),
#         tokenizer=processor.tokenizer
#     )

#     # Extract tokens generated beyond input
#     generated_tokens = output[0, inputs['input_ids'].size(1):]
#     response = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

#     return response
#     # except Exception as e:
#     #     print(f"Error during inference: {e}")
#     #     return None

# # -------------------------
# # Example usage
# # -------------------------
# if __name__ == "__main__":
#     image_input = "/projects/data/vision-team/shanmukha_sreevatsa/IMC/tb3.png"  # local image path
#     question = "Who issued this notice?"
#     answer = get_patram_response(image_input, question)

#     if answer:
#         print("Answer:", answer)
#     else:
#         print("No answer generated.")

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
from PIL import Image
import requests

# Model ID and device setup
# model_id = "bharatgenai/patram-7b-instruct"
model_id = '/projects/data/vision-team/venkat_kesav/GR_Model_Training_with_Swift/final_outputs/batch_7-14_200k_docs/output_5555/v0-20250708-204650/checkpoint-22004'
device = "cuda:7" if torch.cuda.is_available() else "cpu"
device = 'cuda:7'

# Load processor and model
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True
).to(device)

def get_patram_response(image_path_or_url, question):
    try:
        # Load image
        if image_path_or_url.startswith("http"):
            image = Image.open(requests.get(image_path_or_url, stream=True).raw).convert("RGB")
        else:
            image = Image.open(image_path_or_url).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # Format the prompt as expected
    prompt = f"Question: {question} Answer based on the image."

    try:
        # Preprocess image and text using the processor
        inputs = processor.process(images=[image], text=prompt)
        inputs = {k: v.to(device).unsqueeze(0) for k, v in inputs.items()}

        # Generate output using model's generate_from_batch method (Patram-specific)
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

        # Extract generated tokens (excluding input tokens) and decode
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        response = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return response
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

# Example usage:
image_input = "https://knowscope.in/wp-content/uploads/2025/05/cghd-nag.png"
question = "Who issued this notice?"
answer = get_patram_response(image_input, question)
if answer:
    print("Answer:", answer)
