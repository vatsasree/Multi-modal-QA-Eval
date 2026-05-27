import argparse
import logging
import json
import time
import torch
import tqdm
from PIL import Image
from typing import Optional, Dict, Any, List, Tuple
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run batch Patram inference over a QA dataset.")
    parser.add_argument("--model_id", type=str, required=True,
                        help="Hugging Face model ID or local path to Patram model")
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to the input JSON file containing QA pairs")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Path to save the output JSON file with timed answers")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--use_bfloat16", action="store_true",
                        help="Use bfloat16 precision (recommended for Ampere/Hopper GPUs)")
    return parser.parse_args()

def load_model_and_processor(model_id: str, use_bfloat16: bool) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    """Loads the Patram model and processor."""
    logger.info(f"Loading model: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use bfloat16 for better performance on modern GPUs (e.g., A100, H100)
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
    image_path: str,
    question: str,
    max_new_tokens: int = 200
) -> Optional[str]:
    """Generates an answer from the Patram model given an image and a question."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

    prompt = f"Question: {question} Answer based on the image."
    device = model.device

    try:
        # Preprocess image and text
        inputs = processor.process(images=[image], text=prompt)
        inputs = {k: v.to(device).unsqueeze(0) for k, v in inputs.items()}

        # Generate output
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

        # Decode response
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        response = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return response
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return None

def process_dataset(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    qa_data: List[Dict[str, Any]],
    max_new_tokens: int
) -> List[Dict[str, Any]]:
    """Iterates over the dataset, running inference and recording execution times.
    
    Note: For higher throughput on H100s, consider batching the inputs across 
    documents instead of processing sequentially.
    """
    for doc in tqdm.tqdm(qa_data, desc="Documents"):
        img_path = doc.get('img_path')
        if not img_path:
            logger.warning(f"No image path found for document {doc.get('doc_id')}")
            continue

        qa_pairs = doc.get("qa_pairs", [])
        for qa in tqdm.tqdm(qa_pairs, desc=f"Questions in {doc.get('doc_id', 'unknown')}", leave=False):
            question = qa["question"]

            start_time = time.time()
            answer = get_patram_response(model, processor, img_path, question, max_new_tokens)
            end_time = time.time()

            qa["model1"] = answer if answer else ""
            qa["model1_time"] = round(end_time - start_time, 3)
            
    return qa_data

def main():
    args = parse_args()
    
    try:
        model, processor = load_model_and_processor(args.model_id, args.use_bfloat16)
        
        logger.info(f"Loading dataset from {args.input_json}")
        with open(args.input_json, 'r') as f:
            qa_data = json.load(f)
            
        logger.info("Starting inference loop...")
        updated_data = process_dataset(model, processor, qa_data, args.max_new_tokens)
        
        logger.info(f"Saving updated dataset to {args.output_json}")
        with open(args.output_json, "w") as f:
            json.dump(updated_data, f, indent=4)
            
        logger.info("Processing complete.")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")

if __name__ == "__main__":
    main()