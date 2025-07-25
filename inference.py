import joblib
import numpy as np
import pandas as pd
save_dir = r'D:\Scraper\model'
models = joblib.load(f"{save_dir}/cs_best_models.joblib")
label_encoders = joblib.load(f"{save_dir}/cs_label_encoders.joblib")
embedder = joblib.load(f"{save_dir}/cs_embedder.joblib")
target_columns = ['Sub Kategori', 'Kategori', 'Sub Askes']

def predict_single(text):
    vec = embedder.encode([text])
    pred = []
    for col in target_columns:
        y_pred = models[col].predict(vec)
        label = label_encoders[col].inverse_transform(y_pred)[0]
        pred.append(label)
    return dict(zip(target_columns, pred))

def predict_batch(texts):
    vecs = embedder.encode(texts)
    results = {col: [] for col in target_columns}
    for col in target_columns:
        y_preds = models[col].predict(vecs)
        labels = label_encoders[col].inverse_transform(y_preds)
        results[col] = labels
    df_out = pd.DataFrame(results)
    df_out['Pengaduan'] = texts
    return df_out
