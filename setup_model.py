import os
import shutil
import pickle
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import pandas as pd
from clustering.preprocessing import proses_upload_data
from clustering.kmeans_module import jalankan_kmeans

# 1. Process dataset
df = proses_upload_data('uploads')

model_folder = 'models/model_utama'
os.makedirs(model_folder, exist_ok=True)

# 2. Run K-Means
jalankan_kmeans(df, save_path=model_folder)
with open(os.path.join(model_folder, "data_gabungan_clean.pkl"), "wb") as f:
    pickle.dump(df, f)

# 3. Create meta.json
meta = {
    "nama_data": "Dataset Mhs Polinema 1",
    "uploaded_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(timespec="seconds"),
}
with open(os.path.join(model_folder, "meta.json"), "w") as f:
    json.dump(meta, f)

# 4. Set active model
with open("models/active_model.txt", "w") as f:
    f.write("model_utama")

# 5. Delete other models
for d in os.listdir('models'):
    if d.startswith('model_') and d != 'model_utama':
        try:
            shutil.rmtree(os.path.join('models', d))
            print(f"Deleted {d}")
        except:
            pass

# 6. Delete FCM module
if os.path.exists('clustering/fcm_module.py'):
    os.remove('clustering/fcm_module.py')
    print("Deleted fcm_module.py")

print("Setup completed!")
