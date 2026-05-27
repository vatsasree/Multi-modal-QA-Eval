import argparse
import logging
import torch
from typing import Optional, List, Dict, Any
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL model inference.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Hugging Face model ID or local path")
    parser.add_argument("--image_url", type=str, 
                        default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                        help="URL or local path to the image")
    parser.add_argument("--prompt", type=str, default="Describe the image in detail.",
                        help="Text prompt for the model")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--use_flash_attention", action="store_true",
                        help="Use flash_attention_2 for faster inference (requires Ampere+ GPU)")
    return parser.parse_args()

def load_model_and_processor(model_id: str, use_flash_attention: bool) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    """Loads the Qwen VLM and its processor with performance optimizations."""
    logger.info(f"Loading model: {model_id}")
    
    # Use bfloat16 for better performance on modern GPUs (e.g., A100, H100)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    attn_implementation = "flash_attention_2" if use_flash_attention else "sdpa"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=attn_implementation
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def run_inference(
    model: Qwen2_5_VLForConditionalGeneration, 
    processor: AutoProcessor, 
    image_path: str, 
    prompt: str, 
    max_new_tokens: int
) -> str:
    """Runs inference on a single image and text prompt."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    logger.info("Preparing inputs...")
    inputs = processor.apply_chat_template(
        conversation,
        fps=1,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    logger.info("Generating output...")
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # Exclude prompt tokens from the output
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    return output_text[0]

def main():
    args = parse_args()
    
    try:
        model, processor = load_model_and_processor(args.model_id, args.use_flash_attention)
        response = run_inference(model, processor, args.image_url, args.prompt, args.max_new_tokens)
        
        print("\n" + "="*50)
        print(f"Prompt: {args.prompt}")
        print("="*50)
        print(f"Response:\n{response}")
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")

if __name__ == "__main__":
    main()