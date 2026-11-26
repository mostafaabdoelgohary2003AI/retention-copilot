"""Download the public data and models used by the AI modules."""
from huggingface_hub import hf_hub_download, snapshot_download
import shutil
p = hf_hub_download("ar_res_reviews", "data/train-00000-of-00001.parquet", repo_type="dataset")
shutil.copy(p, "data/ar_res_reviews.parquet")
snapshot_download("CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment", allow_patterns=["*.json","*.txt","*.safetensors"])
snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", allow_patterns=["*.json","*.txt","*.safetensors"])
print("done")
