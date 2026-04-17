from huggingface_hub import hf_hub_download
import os

def download_weights():
    os.makedirs("weights", exist_ok=True)
    os.makedirs("configs", exist_ok=True)

    if not os.path.exists("weights/best_model.pt"):
        print("Downloading model weights...")
        hf_hub_download(
            repo_id="annehakobyan/plant-disease-classifier",
            filename="best_model.pt",
            local_dir="weights"
        )

    if not os.path.exists("configs/class_map.json"):
        print("Downloading class map...")
        hf_hub_download(
            repo_id="annehakobyan/plant-disease-classifier",
            filename="class_map.json",
            local_dir="configs"
        )

if __name__ == "__main__":
    download_weights()