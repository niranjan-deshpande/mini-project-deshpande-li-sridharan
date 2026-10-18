from huggingface_hub import snapshot_download
p = snapshot_download("openai/gpt-oss-120b", allow_patterns=["*.json", "*.jinja", "model-*.safetensors"], max_workers=4)
print("DONE", p)
