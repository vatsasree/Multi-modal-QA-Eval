import os
import io
import fitz  # PyMuPDF
import torch
from PIL import Image
import streamlit as st
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig

# -------------------
# HF Cache + Token
# -------------------
cache_root = "/projects/data/vision-team/shanmukha_sreevatsa/hf_cache"
os.environ["HF_HOME"] = cache_root
os.environ["HF_HUB_CACHE"] = cache_root
os.environ["TRANSFORMERS_CACHE"] = cache_root

hf_token = os.getenv("HF_TOKEN")

# -------------------
# Model setup
# -------------------
model_id = "/projects/data/vision-team/venkat_kesav/GR_Model_Training_with_Swift/final_outputs/batch_7-14_200k_docs/output_5555/v0-20250708-204650/checkpoint-22004"
device = "cuda:7" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if "cuda" in device else torch.float32

# Load processor and model (once, globally)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch_dtype).to(device)

# -------------------
# Core Inference
# -------------------
def get_patram_response(file, question: str):
    if file is None:
        return "⚠️ Please upload an image or a PDF."

    # Handle PDF
    if file.name.lower().endswith(".pdf"):
        try:
            doc = fitz.open(stream=file.read(), filetype="pdf")
            if len(doc) == 0:
                return "⚠️ PDF is empty."
            pix = doc[0].get_pixmap()
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        except Exception as e:
            return f"⚠️ Failed to process PDF: {e}"
    else:
        try:
            image = Image.open(file).convert("RGB")
        except Exception as e:
            return f"⚠️ Failed to open image: {e}"

    # Use fallback prompt
    if not question:
        question = "Describe the document."

    prompt = f"Question: {question} Answer based on the image."

    try:
        # Preprocess
        inputs = processor.process(images=[image], text=prompt)
        inputs = {k: v.to(device).unsqueeze(0) for k, v in inputs.items()}

        # Generate
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=300, stop_strings=["<|endoftext|>"]),
            tokenizer=processor.tokenizer
        )

        # Decode
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        response = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        return response if response else "⚠️ Model returned empty response."
    except Exception as e:
        return f"⚠️ Error during inference: {e}"

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Patram Latest Document Vision Chat", layout="wide")
st.title("📄 Patram: Document Vision Chat")

st.write("Upload an image or PDF and ask a question. The model will respond based on the document content.")

uploaded_file = st.file_uploader("Upload image or PDF", type=["png", "jpg", "jpeg", "pdf"])
question = st.text_input("Enter your question", value="Describe the document.")

if st.button("Run Inference"):
    with st.spinner("Running model..."):
        answer = get_patram_response(uploaded_file, question)
        st.success("✅ Done!")
        st.text_area("Model Output", value=answer, height=200)

    # Display image preview (skip for PDFs)
    if uploaded_file and uploaded_file.type.startswith("image/"):
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)