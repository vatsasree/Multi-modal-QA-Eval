import os
# come before ANY library that touches huggingface_hub 
cache_root = "/projects/data/vision-team/shanmukha_sreevatsa/hf_cache"
os.environ["HF_HOME"] = cache_root          # root for every HF cache
os.environ["HF_HUB_CACHE"] = cache_root     # hub-specific override
os.environ["TRANSFORMERS_CACHE"] = cache_root   # models only (older var)


import gradio as gr
import fitz  # PyMuPDF
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import torch
import io
import torch

# Add HF_Token and Cache path. 
hf_token = os.getenv("HF_TOKEN")

# Detect device
if torch.cuda.is_available():
    device_map = {"": 7} # auto" # {"": 4}
    torch_dtype = torch.float16  #Use FP16 for GPUs
else:
    device_map = {"": "cpu"}     # Don't use auto on CPU
    torch_dtype = torch.float32  # Use FP32 for CPU (safer)

# Load processor and model (once, globally)
processor = AutoProcessor.from_pretrained(
    '/projects/data/vision-team/venkat_kesav/GR_Model_Training_with_Swift/final_outputs/batch_7-14_200k_docs/output_5555/v0-20250708-204650/checkpoint-22004',
    trust_remote_code=True,
    device_map = device_map,
)

model = AutoModelForCausalLM.from_pretrained(
    '/projects/data/vision-team/venkat_kesav/GR_Model_Training_with_Swift/final_outputs/batch_7-14_200k_docs/output_5555/v0-20250708-204650/checkpoint-22004',
    trust_remote_code=True,
    device_map = device_map,
)

# Core inference function
def process_image_and_prompt(file, prompt):
    if file is None:
        return "⚠️ Please upload an image or a PDF document."

    # Convert PDF to image if it's a PDF
    if file.name.lower().endswith(".pdf"):
        doc = fitz.open(file.name)
        if len(doc) == 0:
            return "⚠️ PDF is empty."
        # Render the first page to image
        pix = doc[0].get_pixmap()
        image = Image.open(io.BytesIO(pix.tobytes("png")))
    else:
        image = Image.open(file)

    # Use fallback prompt if none given
    if not prompt:
        prompt = "Describe the document."

    # Prepare input for model
    inputs = processor.process(images=[image], text=prompt)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=300, stop_strings=["<|endoftext|>"]),
        tokenizer=processor.tokenizer
    )

    # Decode generated text
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text.strip()


# Gradio Interface
demo = gr.Interface(
    fn=process_image_and_prompt,
    inputs=[
        gr.File(label="Upload image or PDF", file_types=[".png", ".jpg", ".jpeg", ".pdf"]),
        gr.Textbox(label="Prompt (optional, default: 'Describe the document')")
    ],
    outputs=gr.Textbox(label="Model Output"),
    title="Patram Latest Document Vision Chat",
    description="Upload an image or a single-page PDF and optionally give a prompt. The model will describe or answer based on visual content."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)   # public URL
