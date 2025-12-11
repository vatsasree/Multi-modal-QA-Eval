
# Multi-modal-QA-Eval 📊

A comprehensive framework for evaluating state-of-the-art Vision-Language Models (VLMs) on multi-lingual Indic document Question Answering (QA). 

This repository houses the benchmarking scripts and inference pipelines used to contrast standard **Image QA** approaches against a novel **Semantic-HTML** pipeline.

## 📝 Abstract

This project evaluates models like **Qwen**, **Gemma**, and **DeepSeek** on their ability to understand and reason over Indic document images. It establishes a comparative framework between:
1.  **Direct Image QA:** Feeding raw document images directly to VLMs.
2.  **Semantic-HTML QA:** A pipeline where documents are first reconstructed into high-fidelity HTML (via a Visual-to-HTML system) and then queried using Large Language Models (e.g., GPT-OSS-120B).

## ✨ Features

* **Multi-Model Support:** Scripts to run inference on Qwen, Gemma, DeepSeek, and Patram models.
* **Dual-Modality Evaluation:** Compare performance between visual inputs (pixels) and semantic inputs (HTML code).
* **Containerized Environment:** Full Docker support for reproducible evaluation environments.
* **Interactive Demos:** Includes Streamlit/Gradio scripts for local testing and visualization.
* **Data Utilities:** Tools for caption generation (`generate_captions.py`) and format conversion (`csv2json.py`).

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `patram_inference.py` | Inference pipeline specifically for the **Patram** document model. |
| `qwen.py` | Inference script for Qwen-family VLMs. |
| `inference_oss.py` | General inference wrapper for Open Source models (OSS). |
| `streamlit_patram.py` | Interactive Streamlit dashboard for testing model outputs. |
| `generate_captions.py` | Utility to generate synthetic captions or QA pairs from data. |
| `Dockerfile` & `run_docker.sh` | Container configuration and startup scripts for deployment. |
| `requirements.txt` | Python dependencies. |

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* NVIDIA GPU (Recommended for VLM inference)
* Docker (Optional, for containerized run)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Sreevatsa/Multi-modal-QA-Eval.git](https://github.com/Sreevatsa/Multi-modal-QA-Eval.git)
    cd Multi-modal-QA-Eval
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 🐳 Docker Usage

To run the evaluation suite in an isolated container (recommended for H100/A100 clusters):

```bash
# Build the image
docker build -t multimodal-eval .

# Run the container using the provided helper script
bash run_docker.sh
````

## 🛠️ Usage

### Running Inference

You can run specific model evaluations using the python scripts provided.

**For Patram Model:**

```bash
python patram_inference.py --input_path /path/to/images --model_path /path/to/patram
```

**For Qwen Model:**

```bash
python qwen.py --model_name Qwen/Qwen-VL-Chat --image_file example_doc.jpg
```

### Launching the Demo

To visualize results locally using the Streamlit interface:

```bash
streamlit run streamlit_patram.py
```

## 📊 Methodology

The core experiment contrasts two pipelines:

1.  **VLM (Image-based):** `Image + Prompt -> [VLM] -> Answer`
2.  **HTML-LLM (Text-based):** `Image -> [Visual-to-HTML] -> HTML Code + Prompt -> [LLM] -> Answer`

*Evaluation metrics include standard QA accuracy, hallucination rate, and structural fidelity.*
