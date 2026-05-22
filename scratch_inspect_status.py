import pickle
import os

pkl_path = "models/model_25/hasil_kmeans_3cluster.pkl"
with open(pkl_path, "rb") as f:
    df = pickle.load(f)["data"]

if "STATUS" in df.columns:
    print("STATUS value counts:")
    print(df["STATUS"].value_counts(dropna=False))
else:
    print("STATUS column not found")
