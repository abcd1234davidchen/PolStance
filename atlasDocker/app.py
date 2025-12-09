from huggingface_hub import hf_hub_download
import os

hf_hub_download(repo_id="TWCKaijin/PolStance",filename="embeddings.parquet", repo_type="dataset",local_dir=".",local_dir_use_symlinks=False, token=os.getenv("HF_TOKEN",None))