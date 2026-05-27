import argparse
import logging
from typing import Optional
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run inference using GPT-OSS style models.")
    parser.add_argument("--model_id", type=str, default="openai/gpt-oss-20b",
                        help="Hugging Face model ID")
    parser.add_argument("--html_file", type=str, default=None,
                        help="Path to an HTML file to use as context")
    parser.add_argument("--html_content", type=str, default=None,
                        help="Raw HTML string to use as context (if html_file is not provided)")
    parser.add_argument("--question", type=str, default="Summarize the HTML",
                        help="Question to ask based on the HTML")
    parser.add_argument("--max_new_tokens", type=int, default=1000,
                        help="Maximum number of tokens to generate")
    return parser.parse_args()

def load_pipeline(model_id: str):
    """Loads the text-generation pipeline."""
    logger.info(f"Loading text-generation pipeline for model: {model_id}")
    # bfloat16 is highly recommended for modern GPUs like H100
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype="bfloat16",
        device_map="auto",
    )
    return pipe

def get_html_context(html_file: Optional[str], html_content: Optional[str]) -> str:
    """Retrieves HTML context from a file or raw string."""
    if html_file:
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read HTML file {html_file}: {e}")
            raise
    elif html_content:
        return html_content
    else:
        # Default fallback
        return """
        ```html
        <html><body>
        <p style="position: absolute; left: 13px; top: 7px; width: 979px; height: 197px;">Expression of interest/ applications are invited for Senior Consultants (02 posts) on Sustainable<br/>Land Management on contractual basis in Centre of Excellence on Sustainable Land<br/>Management, Indian Council of Forestry Research and Education, Dehradun on payment of<br/>consolidated monthly remuneration/ fee of Rs. 1,25,000/-. Full details of the advertisement are<br/>available on ICFRE website (www.icfre.gov.in) under the link recruitment.</p>
        </body></html>
        ```
        """

def run_inference(pipe, html_context: str, question: str, max_new_tokens: int) -> str:
    """Runs the model pipeline with the given HTML context and question."""
    prompt = f"""Given the following HTML content:
{html_context}

Answer the following question:
Question: {question}
"""
    logger.info("Generating response...")
    outputs = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        return_full_text=False,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    
    return outputs[0]["generated_text"].strip()

def main():
    args = parse_args()
    
    try:
        pipe = load_pipeline(args.model_id)
        html_context = get_html_context(args.html_file, args.html_content)
        
        answer = run_inference(pipe, html_context, args.question, args.max_new_tokens)
        
        print("\n" + "="*50)
        print(f"Question: {args.question}")
        print("="*50)
        print(f"Answer:\n{answer}")
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")

if __name__ == "__main__":
    main()
