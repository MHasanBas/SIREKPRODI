import os

def load_active_model_name() -> str:
    active_flag_path = "models/active_model.txt"
    if os.path.exists(active_flag_path):
        with open(active_flag_path, "r") as f:
            return f.read().strip() or "model_utama"
    return "model_utama"

def get_next_model_folder() -> str:
    base_dir = "models"
    existing = [d for d in os.listdir(base_dir) if d.startswith("model_") and d != "model_utama"]
    max_id = 0
    for name in existing:
        try:
            num = int(name.split("_", 1)[1])
            max_id = max(max_id, num)
        except (IndexError, ValueError):
            continue
    next_id = max_id + 1
    folder_path = os.path.join(base_dir, f"model_{next_id}")
    os.makedirs(folder_path, exist_ok=True)
    return folder_path
