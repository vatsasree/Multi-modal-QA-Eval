# testing script for gpt-oss

# from transformers import pipeline
# import torch

# model_id = "openai/gpt-oss-20b"

# # Example: Use GPU 0
# device = 7 # change to the GPU number you want, e.g., 1, 2, etc.

# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     torch_dtype=torch.float16,  # usually 'auto' picks best dtype; float16 for efficiency on GPU
#     device=device,  # single GPU index
# )

# messages = [
#     {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
# ]

# outputs = pipe(
#     messages,
#     max_new_tokens=256,
# )

# print(outputs[0]["generated_text"])


from transformers import pipeline

model_id = "openai/gpt-oss-20b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="bfloat16",  # important for H100
    device_map="auto",       # let HF spread model across GPUs if needed
)
html = """
```html
<html><body>
<p style="position: absolute; left: 13px; top: 7px; width: 979px; height: 197px;">Expression of interest/ applications are invited for Senior Consultants (02 posts) on Sustainable<br/>Land Management on contractual basis in Centre of Excellence on Sustainable Land<br/>Management, Indian Council of Forestry Research and Education, Dehradun on payment of<br/>consolidated monthly remuneration/ fee of Rs. 1,25,000/-. Full details of the advertisement are<br/>available on ICFRE website (www.icfre.gov.in) under the link recruitment.</p>
</body></html>
```
"""
question = "Summarize the HTML"

prompt = f"""Given a HTML {html}. Answer the following question. 
Question: {question}
"""

# outputs = pipe(prompt, max_new_tokens=1000, return_full_text=False)
# print(outputs[0]["generated_text"])

outputs = pipe(
    prompt,
    max_new_tokens=1000,
    return_full_text=False,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)

summary = outputs[0]["generated_text"].strip()
print(summary)
