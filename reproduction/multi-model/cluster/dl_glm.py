from huggingface_hub import snapshot_download
p = snapshot_download("zai-org/GLM-4.5-Air", allow_patterns=["*.json", "*.jinja", "model-*.safetensors"], max_workers=8)
print("DONE", p)
