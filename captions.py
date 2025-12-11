import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from pathlib import Path
from bs4 import BeautifulSoup
import base64
import requests
from io import BytesIO
from PIL import Image

# -------------------------
# Configuration
# -------------------------
INPUT_HTML_FOLDER = Path("/projects/data/vision-team/aryan_jain/BGenV-Data_viewer/IMC_data")
PROMPT_TEMPLATE = "Describe this image. Give a detailed caption for the image. The caption should be very detailed such that any LLM without even looking at the image should be able to understand all the nuances of the image."
MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"
MAX_NEW_TOKENS = 1000

# -------------------------
# Load HF Qwen2.5VL Model
# -------------------------
print("Loading Qwen2.5VL model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# -------------------------
# Helper Functions
# -------------------------
def url_or_base64_to_image(url_or_base64: str) -> Image.Image:
    """Load image from URL or base64 string."""
    if url_or_base64.startswith("data:image/"):
        # base64
        header, encoded = url_or_base64.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        return Image.open(BytesIO(img_bytes)).convert("RGB")
    else:
        # URL
        resp = requests.get(url_or_base64)
        return Image.open(BytesIO(resp.content)).convert("RGB")

def generate_caption_hf(image: Image.Image, prompt: str = PROMPT_TEMPLATE) -> str:
    """Generate caption using HF Qwen2.5VL model."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        fps=1,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

# def process_html_file(html_path: Path):
#     """Update HTML: replace base64 images in <body> with alt text captions."""
#     html_content = html_path.read_text(encoding="utf-8")
#     soup = BeautifulSoup(html_content, "html.parser")

#     if not soup.body:
#         print(f"No <body> found in {html_path.name}")
#         return

#     img_tags = soup.body.find_all("img")
#     if not img_tags:
#         print(f"No images found in <body> of {html_path.name}")
#         return

#     for idx, img_tag in enumerate(img_tags, 1):
#         src = img_tag.get("src", "")
#         if src:
#             image = url_or_base64_to_image(src)
#             caption = generate_caption_hf(image)
#             print(f"Processed image {idx} in {html_path.name}: {caption}")

#             # Update <img> tag
#             img_tag["alt"] = caption
#             img_tag["src"] = ""  # strip base64 (optional)

#     # Save new HTML
#     output_path = html_path.with_name(html_path.stem + "_caption.html")
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(str(soup))
#     print(f"Saved updated HTML to {output_path}")
#     print(str(soup))

def process_html_file(html_path: Path):
    """Update HTML: replace base64 images in <body> with alt text captions, always save new HTML."""
    html_content = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html_content, "html.parser")

    if not soup.body:
        print(f"No <body> found in {html_path.name}")
    else:
        img_tags = soup.body.find_all("img")
        if not img_tags:
            print(f"No images found in <body> of {html_path.name}")
        else:
            for idx, img_tag in enumerate(img_tags, 1):
                src = img_tag.get("src", "")
                if src:
                    image = url_or_base64_to_image(src)
                    caption = generate_caption_hf(image)
                    print(f"Processed image {idx} in {html_path.name}: {caption}")

                    # Update <img> tag
                    img_tag["alt"] = caption
                    img_tag["src"] = ""  # strip base64 (optional)

    # Always save the HTML, even if no images or body
    output_path = html_path.with_name(html_path.stem + "_caption.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(soup))
    print(f"Saved updated HTML to {output_path}")


# -------------------------
# Main
# -------------------------
def main():
    html_files = list(INPUT_HTML_FOLDER.glob("*/reading_order/*.html"))
    if not html_files:
        print(f"No HTML files found in {INPUT_HTML_FOLDER}")
        return

    for html_file in html_files:
        process_html_file(html_file)

if __name__ == "__main__":
    main()
