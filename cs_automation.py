# === STEP 1: Install Library jika belum ===
# pip install sentence-transformers scikit-learn pandas joblib tqdm

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')

# === STEP 2: LOAD DATA ===
file_path = r'D:\Scraper\Data Complain dari CS 4K.csv'  # <- adjust path if needed
df = pd.read_csv(file_path)

# Pastikan nama kolom sesuai
target_columns = ['Sub Kategori', 'Kategori', 'Sub Askes']
assert all(col in df.columns for col in target_columns), "Pastikan nama kolom target benar!"

# Bersihkan kolom pengaduan (isi '' jika NaN)
df['Pengaduan'] = df['Pengaduan'].fillna('').astype(str)

# === STEP 3: Load Embedding Model ===
model_name = 'firqaaa/indo-sentence-bert-base'
try:
    embedder = SentenceTransformer(model_name)
except:
    embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

print("Embedding text...")
X = embedder.encode(df['Pengaduan'].tolist(), show_progress_bar=True, batch_size=32)

# === STEP 4: Label Encoding Target ===
label_encoders = {}
y = pd.DataFrame()
for col in target_columns:
    le = LabelEncoder()
    y[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# === STEP 5: Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === STEP 6: Model Training & Evaluation ===
def train_eval_model(model_class, model_params):
    models = {}
    accs = []
    for i, col in enumerate(target_columns):
        clf = model_class(**model_params)
        clf.fit(X_train, y_train[col])
        models[col] = clf
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test[col], y_pred)
        accs.append(acc)
        print(f"{col} acc: {acc:.4f}")
    print(f"Average acc: {np.mean(accs):.4f}")
    return models, np.mean(accs)

print("\nTraining RandomForest...")
rf_models, rf_acc = train_eval_model(RandomForestClassifier, {'n_estimators':100, 'random_state':42, 'n_jobs':-1})

print("\nTraining LogisticRegression...")
lr_models, lr_acc = train_eval_model(LogisticRegression, {'max_iter':1000, 'random_state':42})

# === STEP 7: Select Best Model ===
if rf_acc >= lr_acc:
    best_models = rf_models
    best_name = 'RandomForest'
    best_acc = rf_acc
else:
    best_models = lr_models
    best_name = 'LogisticRegression'
    best_acc = lr_acc

print(f"\nSelected model: {best_name} (avg acc: {best_acc:.4f})")

# === STEP 8: Save Models & Encoders ===
save_dir = r'D:\Scraper\model'
os.makedirs(save_dir, exist_ok=True)

joblib.dump(best_models, os.path.join(save_dir, 'cs_best_models.joblib'))
joblib.dump(label_encoders, os.path.join(save_dir, 'cs_label_encoders.joblib'))
joblib.dump(embedder, os.path.join(save_dir, 'cs_embedder.joblib'))
print("Model, encoder, embedder saved to", save_dir)
