import argparse
import logging
import torch
import requests
from PIL import Image
from typing import Optional, Tuple
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Patram-7B model inference.")
    parser.add_argument("--model_id", type=str, default="bharatgenai/patram-7b-instruct",
                        help="Hugging Face model ID or local path to Patram model")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Local path or URL to the image")
    parser.add_argument("--question", type=str, required=True,
                        help="Question to ask about the image")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--use_bfloat16", action="store_true",
                        help="Use bfloat16 precision (recommended for H100)")
    return parser.parse_args()

def load_model_and_processor(model_id: str, use_bfloat16: bool) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    """Loads the Patram model and processor."""
    logger.info(f"Loading model: {model_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if use_bfloat16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    if device == "cuda" and not use_bfloat16:
        dtype = torch.float16

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto"
    )
    return model, processor

def get_patram_response(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    image_path_or_url: str,
    question: str,
    max_new_tokens: int = 200
) -> Optional[str]:
    """Generates an answer from the Patram model given an image and a question."""
    try:
        # Load image
        if image_path_or_url.startswith("http"):
            image = Image.open(requests.get(image_path_or_url, stream=True).raw).convert("RGB")
        else:
            image = Image.open(image_path_or_url).convert("RGB")
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return None

    # Format the prompt as expected
    prompt = f"Question: {question} Answer based on the image."
    
    device = model.device

    try:
        # Preprocess image and text using the processor
        inputs = processor.process(images=[image], text=prompt)
        # Ensure proper batch dimension and device
        inputs = {k: v.to(device).unsqueeze(0) for k, v in inputs.items()}

        # Generate output using model's generate_from_batch method (Patram-specific)
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

        # Extract generated tokens (excluding input tokens) and decode
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        response = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return response
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return None

def main():
    args = parse_args()
    
    try:
        model, processor = load_model_and_processor(args.model_id, args.use_bfloat16)
        answer = get_patram_response(model, processor, args.image_path, args.question, args.max_new_tokens)
        
        if answer:
            print("\n" + "="*50)
            print(f"Question: {args.question}")
            print("="*50)
            print(f"Answer:\n{answer}")
            print("="*50 + "\n")
        else:
            print("No answer generated.")
            
    except Exception as e:
        logger.error(f"Execution failed: {e}")

if __name__ == "__main__":
    main()
