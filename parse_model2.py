import json
import re
from pathlib import Path

def extract_final_answer(text: str) -> str:
    """
    Extracts the final answer text that appears after 'assistantfinal'.
    Falls back to returning the full text if marker not found.
    """
    if not text:
        return ""
    match = re.search(r'assistantfinal\s*(.*)', text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        # Clean up any quotes or whitespace
        return re.sub(r'^[\'"]|[\'"]$', '', answer).strip()
    return text.strip()

def parse_model2_answers_with_model1(input_path: str, output_path: str):
    """
    Reads the input JSON (with model1 and model2 responses),
    extracts model2 final answers, retains model1,
    and saves the cleaned data to a new JSON file.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    parsed_docs = []
    for doc in data:
        new_doc = {
            "doc_id": doc["doc_id"],
            "img_path": doc["img_path"],
            "qa_pairs": []
        }

        for qa in doc.get("qa_pairs", []):
            final_answer = extract_final_answer(qa.get("model2", ""))
            new_doc["qa_pairs"].append({
                "question": qa.get("question", ""),
                "model1": qa.get("model1", "").strip(),
                "model2_final": final_answer
            })

        parsed_docs.append(new_doc)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_docs, f, indent=2, ensure_ascii=False)

    print(f"✅ Cleaned JSON with model1 + model2_final saved to: {output_path}")

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    input_json = "/projects/data/vision-team/shanmukha_sreevatsa/Patram/qa_pairs_with_answers_oss.json"             # your input file
    output_json = "/projects/data/vision-team/shanmukha_sreevatsa/Patram/qa_pairs_with_answers_oss_parsed.json"      # desired output file
    parse_model2_answers_with_model1(input_json, output_json)
