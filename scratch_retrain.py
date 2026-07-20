import os
import pickle
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from services.model_service import get_next_model_folder
from clustering.kmeans_module import jalankan_kmeans

def main():
    # 1. Load active model's data
    active_model = "model_22"
    data_path = f"models/{active_model}/data_gabungan_clean.pkl"
    print(f"Loading data from {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    df = data["data"] if isinstance(data, dict) else data
    print(f"Loaded {df.shape[0]} rows.")

    # 2. Get next model folder
    new_model_path = get_next_model_folder()
    new_model_name = os.path.basename(new_model_path)
    print(f"Training new model at: {new_model_path} ({new_model_name})")

    # 3. Create initial meta
    meta = {
        "nama_data": "datasettt.xlsx - elbow 5x (Retrained Hybrid Scaling)",
        "uploaded_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(timespec="seconds"),
        "k_selection": "elbow",
    }
    with open(os.path.join(new_model_path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 4. Save data_gabungan_clean.pkl in the new model path
    with open(os.path.join(new_model_path, "data_gabungan_clean.pkl"), "wb") as f:
        pickle.dump(df, f)

    # 5. Run K-Means training with Elbow Method (optimal K)
    print("Starting K-Means + GA training. This might take a moment...")
    jalankan_kmeans(df, n_clusters=None, save_path=new_model_path)

    # 6. Set new model as active
    with open("models/active_model.txt", "w") as f:
        f.write(new_model_name)
    print(f"Success! Model {new_model_name} is now trained and set as active.")

if __name__ == "__main__":
    main()
