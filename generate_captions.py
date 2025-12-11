from pathlib import Path
from bs4 import BeautifulSoup
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# -------------------------
# Configuration
# -------------------------
INPUT_HTML_FOLDER = Path("/projects/data/vision-team/aryan_jain/BGenV-Data_viewer/IMC_data")

PROMPT_TEMPLATE = "Describe this image."
MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# -------------------------
# Load Qwen2.5VL model
# -------------------------
print("Loading Qwen2.5VL model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model.eval()
import pdb; pdb.set_trace()
# -------------------------
# Helper Functions
# -------------------------
def generate_caption_qwen(base64_image, prompt=PROMPT_TEMPLATE):
    """Generate caption for a base64 image using Qwen2.5VL."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image/jpeg;base64,{base64_image}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return output_text[0]

def process_html_file(html_path, output_folder):
    """Update HTML: replace base64 images in <body> with alt text captions."""
    html_content = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html_content, "html.parser")

    if not soup.body:
        print(f"No <body> found in {html_path.name}")
        return

    # Only process <img> tags inside <body>
    img_tags = soup.body.find_all("img")
    if not img_tags:
        print(f"No images found in <body> of {html_path.name}")
        return

    for idx, img_tag in enumerate(img_tags, 1):
        src = img_tag.get("src", "")
        if src.startswith("data:image/"):
            base64_data = src.split(",", 1)[1]
            caption = generate_caption_qwen(base64_data)
            print(f"Processed image {idx} in {html_path.name}: {caption}")

            # Update img tag: remove src, add alt
            img_tag["alt"] = caption
            img_tag["src"] = ""  # optionally save image separately and update src

    # Save updated HTML
    # output_path = output_folder / html_path.name
    output_path = str(html_path).replace('.html', 'caption.html')
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

    for html_file in html_files[:1]:
        process_html_file(html_file, OUTPUT_HTML_FOLDER)

if __name__ == "__main__":
    main()
