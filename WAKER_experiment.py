# ============================================================
# WAKER — Weight Attribution based on Kolmogorov-Arnold
# Networks for Entity Resolution
#
# Experiments on the iTunes-Amazon benchmark, comparing four
# ways of aggregating per-attribute similarities into a
# match/non-match decision:
#
#   (0) NO KAN, NO weights  -> score = mean(sim_i)
#   (1) Manual weights      -> score = sum(sim_i*w_i)/sum(w_i)
#   (2) KAN -> deduce weights (leave-one-out) then weighted score
#   (3) KAN-only classifier -> final = KAN probability
#
# Threshold selection:
#   - For (0)(1)(2): best threshold on VALID for the SCORE
#   - For (3): best threshold on VALID for the KAN probability
#
# Preprocessing:
#   - Text attributes: kept as RAW and CLEAN (accent/case/space
#     normalized) versions; similarity = max(cos(raw), cos(clean))
#   - Numeric/date attributes (Price, Time, Released): parsed with
#     dedicated parsers and scored with exp(-|diff|/scale), where
#     scale is derived from the 75th percentile of TRAIN differences
# ============================================================
import os
import re
import unicodedata
from math import exp
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sentence_transformers import SentenceTransformer, InputExample, losses
from kan import KAN

# ==========================
# 1) CONFIG
# ==========================
os.environ["TRANSFORMERS_CACHE"] = "./huggingface_cache/"
os.makedirs("./huggingface_cache/", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Path to the folder containing tableA.csv, tableB.csv, train.csv,
# valid.csv, test.csv for the dataset (e.g. iTunes-Amazon).
# Override without touching the code:
#   Windows (PowerShell):  $env:KAN_ER_DATA_DIR="D:\data\iTunes"
#   Linux/Mac:              export KAN_ER_DATA_DIR=/path/to/iTunes
base_path = os.environ.get("KAN_ER_DATA_DIR", r"C:\Users\lbouabdelli\Experiment\iTunes")

# ==========================
# 2) LOAD DATA
# ==========================
tableA = pd.read_csv(os.path.join(base_path, "tableA.csv")).reset_index(drop=True)
tableB = pd.read_csv(os.path.join(base_path, "tableB.csv")).reset_index(drop=True)
train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
valid_df = pd.read_csv(os.path.join(base_path, "valid.csv"))
test_df = pd.read_csv(os.path.join(base_path, "test.csv"))

# ==========================
# 3) ATTRIBUTES
# ==========================
attribute_list = [
    "Song_Name",
    "Artist_Name",
    "Album_Name",
    "Genre",
    "Price",
    "CopyRight",
    "Time",
    "Released",
]

missingA = [c for c in attribute_list if c not in tableA.columns]
missingB = [c for c in attribute_list if c not in tableB.columns]
if missingA or missingB:
    raise ValueError(f"Missing columns. tableA: {missingA} | tableB: {missingB}")

# ============================================================
# 3bis) PREPROCESSING / STANDARDIZATION (RAW + CLEAN)
#   - text: keep raw + clean
#   - numeric/date: parsed later (Price, Time, Released)
# ============================================================
TEXT_ATTRS = ["Song_Name", "Artist_Name", "Album_Name", "Genre", "CopyRight"]
NUM_ATTRS = ["Price", "Time", "Released"]  # handled with dedicated parsing+similarity


def normalize_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


def basic_clean(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = normalize_unicode(s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_text_keep_some_symbols(x) -> str:
    # keep digits + letters + spaces + a few separators
    s = basic_clean(x)
    s = re.sub(r"[^a-z0-9\s\-\&\.\'/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def preprocess_keep_raw_clean(df: pd.DataFrame, text_attrs: list) -> pd.DataFrame:
    df = df.copy()
    for a in text_attrs:
        df[a] = df[a].fillna("").astype(str)
        df[a + "_raw"] = df[a].astype(str).str.strip()
        df[a + "_clean"] = df[a].apply(clean_text_keep_some_symbols)
    return df


print("\n>>> Preprocessing tableA/tableB (RAW + CLEAN for text)...")
tableA = preprocess_keep_raw_clean(tableA, TEXT_ATTRS)
tableB = preprocess_keep_raw_clean(tableB, TEXT_ATTRS)
print(">>> Preprocessing done.")


# ============================================================
# 4) SBERT pair text for fine-tune (use CLEAN for stability)
# ============================================================
def build_concat_text(rec: pd.Series) -> str:
    parts = []
    for a in attribute_list:
        if a in TEXT_ATTRS:
            parts.append(f"{a}: {rec.get(a + '_clean', '')}")
        else:
            parts.append(f"{a}: {rec.get(a, '')}")
    return " [SEP] ".join(parts)


# ======================================
# 5) SBERT TRAIN EXAMPLES
# ======================================
train_examples = []
for _, row in train_df.iterrows():
    recA = tableA.iloc[int(row["ltable_id"])]
    recB = tableB.iloc[int(row["rtable_id"])]
    train_examples.append(
        InputExample(
            texts=[build_concat_text(recA), build_concat_text(recB)],
            label=int(row["label"]),
        )
    )

# ==============================
# 6) Fine-tune MiniLM (optional)
# ==============================
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name, cache_folder="./huggingface_cache/").to(device)

DO_FINETUNE = True
if DO_FINETUNE:
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=2,
    )
    print("\n>>> Fine-tuning MiniLM...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=5,
        warmup_steps=int(len(train_dataloader) * 0.1),
        show_progress_bar=False,
    )
else:
    print("\n>>> Skipping fine-tune.")

# ============================================================
# 7) PRECOMPUTE EMBEDDINGS (TEXT ONLY, raw + clean)
# ============================================================
def precompute_table_embeddings(st_model, table, cols, batch_size=128, device="cpu"):
    embs = {}
    for c in cols:
        print(f"Precomputing embeddings for '{c}'...")
        texts = table[c].fillna("").astype(str).tolist()
        vecs = st_model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        embs[c] = torch.tensor(vecs, dtype=torch.float32, device=device)
    return embs


text_embed_cols = []
for a in TEXT_ATTRS:
    text_embed_cols.append(a + "_raw")
    text_embed_cols.append(a + "_clean")

print("\n>>> Precomputing embeddings for tableA & tableB (TEXT raw+clean)...")
tableA_embs = precompute_table_embeddings(model, tableA, text_embed_cols, device=device)
tableB_embs = precompute_table_embeddings(model, tableB, text_embed_cols, device=device)
print(">>> Done.")


# ============================================================
# 8) Dedicated parsers + similarities for numeric/date attrs
# ============================================================
def try_parse_price(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if not s:
        return np.nan
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9\.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return np.nan


def try_parse_time_to_seconds(x):
    # supports "mm:ss", "m:ss", "ss", "240", etc.
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if not s:
        return np.nan
    s = s.replace(" ", "")
    # mm:ss
    if re.match(r"^\d+:\d{1,2}$", s):
        m, sec = s.split(":")
        try:
            return int(m) * 60 + int(sec)
        except Exception:
            return np.nan
    # numeric seconds
    s2 = re.sub(r"[^0-9\.\-]", "", s)
    try:
        return float(s2)
    except Exception:
        return np.nan


def try_parse_year_or_date(x):
    # returns year as float if possible (keeps it simple/robust)
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    # year like 2008
    m = re.search(r"(19\d{2}|20\d{2})", s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    # try iso date
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(s[:10], fmt)
            return float(dt.year)
        except Exception:
            continue
    return np.nan


def sim_numeric(a, b, scale):
    if np.isnan(a) or np.isnan(b):
        return 0.0
    return float(exp(-abs(a - b) / max(scale, 1e-6)))


def compute_scale_from_train(attr: str, parser_fn) -> float:
    a_vals = tableA.loc[train_df["ltable_id"], attr].apply(parser_fn)
    b_vals = tableB.loc[train_df["rtable_id"], attr].apply(parser_fn)
    diffs = (a_vals - b_vals).abs()
    diffs = diffs[~diffs.isna()]
    if len(diffs) < 50:
        return 1.0
    return float(np.percentile(diffs.values, 75)) + 1e-6


print("\n>>> Computing numeric scales from TRAIN...")
scale_price = compute_scale_from_train("Price", try_parse_price)
scale_time = compute_scale_from_train("Time", try_parse_time_to_seconds)
scale_released = compute_scale_from_train("Released", try_parse_year_or_date)
print("scale_price   =", scale_price)
print("scale_time    =", scale_time)
print("scale_released=", scale_released)


# ============================================================
# 9) Similarity vector (8 dims, aligned with attribute_list)
#   - text attrs: max(cos(raw), cos(clean)) then map to [0,1]
#   - numeric/date attrs: exp(-|diff|/scale) in [0,1]
# ============================================================
@torch.no_grad()
def sim_text_attr(attr: str, idxA: int, idxB: int) -> float:
    a_raw = tableA_embs[attr + "_raw"][idxA]
    b_raw = tableB_embs[attr + "_raw"][idxB]
    s1 = F.cosine_similarity(a_raw, b_raw, dim=0).item()
    a_cl = tableA_embs[attr + "_clean"][idxA]
    b_cl = tableB_embs[attr + "_clean"][idxB]
    s2 = F.cosine_similarity(a_cl, b_cl, dim=0).item()
    # cosine in [-1,1] -> map to [0,1]
    s = max(s1, s2)
    s01 = (s + 1.0) / 2.0
    return float(np.clip(s01, 0.0, 1.0))


@torch.no_grad()
def extract_similarity_vector_by_index(idxA: int, idxB: int) -> np.ndarray:
    sims = []
    for attr in attribute_list:
        if attr in TEXT_ATTRS:
            sims.append(sim_text_attr(attr, idxA, idxB))
        elif attr == "Price":
            a = try_parse_price(tableA.loc[idxA, "Price"])
            b = try_parse_price(tableB.loc[idxB, "Price"])
            sims.append(sim_numeric(a, b, scale_price))
        elif attr == "Time":
            a = try_parse_time_to_seconds(tableA.loc[idxA, "Time"])
            b = try_parse_time_to_seconds(tableB.loc[idxB, "Time"])
            sims.append(sim_numeric(a, b, scale_time))
        elif attr == "Released":
            a = try_parse_year_or_date(tableA.loc[idxA, "Released"])
            b = try_parse_year_or_date(tableB.loc[idxB, "Released"])
            sims.append(sim_numeric(a, b, scale_released))
        else:
            # fallback (shouldn't happen)
            sims.append(0.0)
    return np.array(sims, dtype=np.float32)


# Build X,y matrices
def build_similarity_matrix(df_pairs: pd.DataFrame):
    X, y = [], []
    for _, row in df_pairs.iterrows():
        X.append(extract_similarity_vector_by_index(int(row["ltable_id"]), int(row["rtable_id"])))
        y.append(int(row["label"]))
    return np.vstack(X).astype(np.float32), np.array(y, dtype=int)


print("\n>>> Building similarity matrices (train/valid/test)...")
X_train, y_train = build_similarity_matrix(train_df)
X_valid, y_valid = build_similarity_matrix(valid_df)
X_test, y_test = build_similarity_matrix(test_df)
print("Shapes:", X_train.shape, X_valid.shape, X_test.shape)


# ==================================================
# 10) Helpers: scoring + threshold tuning + evaluation
# ==================================================
def score_unweighted(sims01: np.ndarray) -> float:
    return float(np.mean(sims01))


def score_weighted(sims01: np.ndarray, weights: dict) -> float:
    num, den = 0.0, 0.0
    for sim, attr in zip(sims01, attribute_list):
        w = float(weights.get(attr, 1.0))
        num += float(sim) * w
        den += w
    return float(num / max(den, 1e-12))


def compute_scores(df_pairs: pd.DataFrame, mode: str, weights: dict = None) -> np.ndarray:
    scores = []
    for _, row in df_pairs.iterrows():
        sims = extract_similarity_vector_by_index(int(row["ltable_id"]), int(row["rtable_id"]))
        if mode == "unweighted":
            scores.append(score_unweighted(sims))
        elif mode == "weighted":
            scores.append(score_weighted(sims, weights))
        else:
            raise ValueError("mode must be 'unweighted' or 'weighted'")
    return np.array(scores, dtype=float)


def best_threshold_from_scores(scores: np.ndarray, y_true: np.ndarray):
    best_thr, best_f1v = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 181):
        pred = (scores >= thr).astype(int)
        f1v = f1_score(y_true, pred, zero_division=0)
        if f1v > best_f1v:
            best_f1v, best_thr = f1v, thr
    return float(best_thr), float(best_f1v)


def eval_scores(scores: np.ndarray, y_true: np.ndarray, thr: float, title: str):
    y_pred = (scores >= thr).astype(int)
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(
        f"thr={thr:.3f} | acc={accuracy_score(y_true,y_pred):.4f} | "
        f"prec={precision_score(y_true,y_pred,zero_division=0):.4f} | "
        f"rec={recall_score(y_true,y_pred,zero_division=0):.4f} | "
        f"f1={f1_score(y_true,y_pred,zero_division=0):.4f}"
    )
    print(classification_report(y_true, y_pred, digits=4))
    return y_pred


# ==================================================
# 11) Train KAN classifier (for (3) and to extract weights)
# ==================================================
torch.manual_seed(42)
np.random.seed(42)

Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
Xva = torch.tensor(X_valid, dtype=torch.float32, device=device)
Xte = torch.tensor(X_test, dtype=torch.float32, device=device)

ytr_np = y_train.astype(int)
yva_np = y_valid.astype(int)
yte_np = y_test.astype(int)
ytr = torch.tensor(ytr_np, dtype=torch.float32, device=device).view(-1, 1)

n_attr = Xtr.shape[1]
kan = KAN(width=[n_attr, 16, 1], grid=7, k=3, seed=42, device=device)

pos = int((ytr_np == 1).sum())
neg = int((ytr_np == 0).sum())
pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


@torch.no_grad()
def probs_from_kan(m: nn.Module, X: torch.Tensor) -> np.ndarray:
    m.eval()
    logits = m(X).view(-1)
    return torch.sigmoid(logits).detach().cpu().numpy()


def best_threshold_from_probs(probs: np.ndarray, y_true: np.ndarray):
    best_thr, best_f1v = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        pred = (probs >= thr).astype(int)
        f1v = f1_score(y_true, pred, zero_division=0)
        if f1v > best_f1v:
            best_f1v, best_thr = f1v, thr
    return float(best_thr), float(best_f1v)


opt_adam = torch.optim.Adam(kan.parameters(), lr=1e-3, weight_decay=1e-4)
print("\n>>> Training KAN (Adam)...")
best_state, best_val_f1 = None, -1.0
for ep in range(1, 51):
    kan.train()
    opt_adam.zero_grad()
    loss = bce(kan(Xtr), ytr)
    loss.backward()
    opt_adam.step()
    if ep % 5 == 0:
        pva = probs_from_kan(kan, Xva)
        thr, f1v = best_threshold_from_probs(pva, yva_np)
        print(f"Epoch {ep:02d} | loss={loss.item():.4f} | valid_best_f1={f1v:.4f} @thr={thr:.2f}")
        if f1v > best_val_f1:
            best_val_f1 = f1v
            best_state = {k: v.detach().cpu().clone() for k, v in kan.state_dict().items()}

if best_state is not None:
    kan.load_state_dict(best_state)

opt_lbfgs = torch.optim.LBFGS(kan.parameters(), lr=1.0, max_iter=250, line_search_fn="strong_wolfe")
print("\n>>> Fine-tuning KAN (LBFGS)...")


def closure():
    opt_lbfgs.zero_grad()
    l = bce(kan(Xtr), ytr)
    l.backward()
    return l


opt_lbfgs.step(closure)


# ==================================================
# 12) Extract KAN weights (leave-one-out) -> used in (2)
# ==================================================
@torch.no_grad()
def leave_one_out_importance(model_kan: nn.Module, X: torch.Tensor) -> np.ndarray:
    model_kan.eval()
    base = model_kan(X).view(-1)  # logits
    imps = np.zeros(X.shape[1], dtype=float)
    for i in range(X.shape[1]):
        Xm = X.clone()
        Xm[:, i] = 0.0
        outm = model_kan(Xm).view(-1)
        imps[i] = torch.mean(torch.abs(base - outm)).item()
    return imps


raw_imp = leave_one_out_importance(kan, Xtr)
s = raw_imp.sum()
norm = (raw_imp / s) if s > 1e-12 else (np.ones_like(raw_imp) / len(raw_imp))
kan_weights_sum1 = {a: float(norm[i]) for i, a in enumerate(attribute_list)}

w_min, w_max = float(norm.min()), float(norm.max())
scaled = np.ones_like(norm) if (w_max - w_min) < 1e-12 else (0.3 + 0.7 * ((norm - w_min) / (w_max - w_min)))
kan_weights_scaled = {a: float(np.round(scaled[i], 4)) for i, a in enumerate(attribute_list)}

print("\n>>> KAN weights (sum=1):")
for a in attribute_list:
    print(f" - {a:12s}: {kan_weights_sum1[a]:.6f}")

print("\n>>> KAN weights (scaled [0.3,1.0]):")
for a in attribute_list:
    print(f" - {a:12s}: {kan_weights_scaled[a]:.4f}")

# ==================================================
# 13) Define MANUAL weights
# ==================================================
manual_weights = {
    "Song_Name": 1.0,
    "Artist_Name": 0.9,
    "Album_Name": 0.9,
    "Genre": 0.4,
    "Price": 0.8,
    "CopyRight": 0.6,
    "Time": 1.0,
    "Released": 0.5,
}

# ==================================================
# 14) RUN 4 EXPERIMENTS
# ==================================================
# (0) NO KAN, NO weights (simple mean)
valid_scores_0 = compute_scores(valid_df, mode="unweighted")
thr0, f10 = best_threshold_from_scores(valid_scores_0, y_valid)
eval_scores(valid_scores_0, y_valid, thr0, "(0) NO KAN, NO weights — VALID (thr tuned on VALID)")

test_scores_0 = compute_scores(test_df, mode="unweighted")
eval_scores(test_scores_0, y_test, thr0, "(0) NO KAN, NO weights — TEST (thr from VALID)")

# (1) Manual weights
valid_scores_1 = compute_scores(valid_df, mode="weighted", weights=manual_weights)
thr1, f11 = best_threshold_from_scores(valid_scores_1, y_valid)
eval_scores(valid_scores_1, y_valid, thr1, "(1) Manual weights — VALID (thr tuned on VALID)")

test_scores_1 = compute_scores(test_df, mode="weighted", weights=manual_weights)
eval_scores(test_scores_1, y_test, thr1, "(1) Manual weights — TEST (thr from VALID)")

# (2) KAN -> deduce weights then weighted score
KAN_WEIGHTS_TO_USE = kan_weights_scaled  # or kan_weights_sum1
valid_scores_2 = compute_scores(valid_df, mode="weighted", weights=KAN_WEIGHTS_TO_USE)
thr2, f12 = best_threshold_from_scores(valid_scores_2, y_valid)
eval_scores(
    valid_scores_2, y_valid, thr2, "(2) KAN-deduced weights + weighted score — VALID (thr tuned on VALID)"
)

test_scores_2 = compute_scores(test_df, mode="weighted", weights=KAN_WEIGHTS_TO_USE)
eval_scores(test_scores_2, y_test, thr2, "(2) KAN-deduced weights + weighted score — TEST (thr from VALID)")

# (3) KAN-only classifier (final decision = KAN prob)
probs_valid = probs_from_kan(kan, Xva)
thr3, f13 = best_threshold_from_probs(probs_valid, yva_np)
pred_valid_3 = (probs_valid >= thr3).astype(int)
print("\n" + "=" * 70)
print("(3) KAN-only classifier — VALID (thr tuned on VALID)")
print("=" * 70)
print(f"thr={thr3:.3f} | acc={accuracy_score(yva_np,pred_valid_3):.4f} | f1={f1_score(yva_np,pred_valid_3,zero_division=0):.4f}")
print(classification_report(yva_np, pred_valid_3, digits=4))

probs_test = probs_from_kan(kan, Xte)
pred_test_3 = (probs_test >= thr3).astype(int)
print("\n" + "=" * 70)
print("(3) KAN-only classifier — TEST (thr from VALID)")
print("=" * 70)
print(f"thr={thr3:.3f} | acc={accuracy_score(yte_np,pred_test_3):.4f} | f1={f1_score(yte_np,pred_test_3,zero_division=0):.4f}")
print(classification_report(yte_np, pred_test_3, digits=4))

# ==================================================
# 15) Quick comparison summary (F1 on VALID)
# ==================================================
print("\n" + "=" * 70)
print("SUMMARY (VALID best F1)")
print("=" * 70)
print(f"(0) NO KAN, NO weights            F1={f10:.4f} @thr={thr0:.3f}")
print(f"(1) Manual weights                F1={f11:.4f} @thr={thr1:.3f}")
print(f"(2) KAN-deduced weights + score   F1={f12:.4f} @thr={thr2:.3f}")
print(f"(3) KAN-only classifier (prob)    F1={f13:.4f} @thr={thr3:.3f}")
print("\nDONE.")
