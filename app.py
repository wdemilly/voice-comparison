# app.py — Voice Comparison Gate
#
# Scores drafts against the Stage 2 voice classifier (Dare/Quinn vs.
# raw pre-edit pipeline drafts). Higher P(human) = prose shape closer
# to Dare/Quinn, further from generator default.
#
# Files that must sit alongside this app.py in the repo root:
#   separation_test.py    (provides extract_pairs_from_text + extract_features)
#   classifier.pkl        (copied from separation_test/output/classifier.pkl)

import streamlit as st
import io
import csv
import os
import pickle
import numpy as np

st.set_page_config(page_title="Voice Comparison", layout="wide")
st.title("Voice Comparison")
st.caption("Stage 2 voice classifier: P(human) = shape-similarity to Dare/Quinn commercial HR register.")

# ---- Load classifier + feature extractor ----
@st.cache_resource
def load_classifier(path="classifier.pkl"):
    if not os.path.exists(path):
        return None, f"classifier.pkl not found at repo root"
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return bundle, None
    except Exception as e:
        return None, f"could not load classifier.pkl: {e}"

bundle, load_err = load_classifier()

try:
    from separation_test import extract_pairs_from_text, extract_features
except Exception as e:
    st.error("Could not import separation_test.py. Make sure it sits at the repo root "
             "alongside app.py.")
    st.code(f"Import error: {e}")
    st.stop()

if bundle is None:
    st.error(load_err)
    st.info("Copy your trained classifier.pkl (from separation_test/output/) to the repo root.")
    st.stop()

# ---- Sidebar ----
st.sidebar.markdown("**Classifier loaded**")
st.sidebar.write(f"Version: `{bundle.get('version', 'unknown')}`")
st.sidebar.write(f"Trained on: {bundle.get('n_human_pairs', '?')} human + "
                 f"{bundle.get('n_ai_pairs', '?')} AI pairs")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Reference scores:**\n"
    "- Dare on *The Duchess Deal*: ~0.77\n"
    "- Quinn on *The Viscount Who Loved Me*: ~0.76\n"
    "- Raw pipeline drafts: ~0.22–0.34\n"
    "- Population gap: ~0.42\n"
    "- CV AUC: 0.82"
)

threshold = st.sidebar.number_input("Voice gate threshold", min_value=0.0, max_value=1.0,
                                     value=0.55, step=0.05)

# ---- Upload ----
uploaded = st.file_uploader("Upload draft(s) (.txt)", type="txt", accept_multiple_files=True)

if not uploaded:
    st.info("Upload one or more .txt drafts. Start with a Dare chapter and a raw pipeline "
            "draft to confirm the anchors come in where they should.")
    st.stop()

# ---- Score each draft ----
def score_text(text, bundle):
    clf = bundle["model"]
    feature_names = bundle["feature_names"]
    pairs = extract_pairs_from_text(text)
    if not pairs:
        return None
    X = np.array([[extract_features(a, b)[f] for f in feature_names]
                  for a, b in pairs], dtype=float)
    probs = clf.predict_proba(X)[:, 1]
    return {
        "n_pairs":       len(pairs),
        "mean_p_human":  float(probs.mean()),
        "median_p":      float(np.median(probs)),
        "frac_above_50": float((probs >= 0.5).mean()),
        "frac_above_70": float((probs >= 0.7).mean()),
    }

rows = []
progress = st.progress(0.0)
status = st.empty()

for i, f in enumerate(uploaded):
    text = f.read().decode("utf-8", errors="ignore")
    try:
        result = score_text(text, bundle)
        if result is None:
            rows.append({"filename": f.name, "mean_p_human": None,
                         "verdict": "ERROR: no sentence pairs extracted"})
        else:
            p = result["mean_p_human"]
            verdict = "PASS" if p >= threshold else "FAIL"
            rows.append({
                "filename":      f.name,
                "mean_p_human":  round(p, 3),
                "median":        round(result["median_p"], 3),
                "frac>=0.5":     round(result["frac_above_50"], 3),
                "frac>=0.7":     round(result["frac_above_70"], 3),
                "n_pairs":       result["n_pairs"],
                "verdict":       verdict,
            })
    except Exception as e:
        rows.append({"filename": f.name, "mean_p_human": None,
                     "verdict": f"ERROR: {e}"})

    progress.progress((i + 1) / len(uploaded))
    status.text(f"Scored {i + 1}/{len(uploaded)}")

# ---- Display ----
valid = sorted([r for r in rows if r.get("mean_p_human") is not None],
               key=lambda r: r["mean_p_human"], reverse=True)
errors = [r for r in rows if r.get("mean_p_human") is None]

st.subheader("Results")
n_pass = sum(1 for r in valid if r["mean_p_human"] >= threshold)
n_fail = len(valid) - n_pass
c1, c2, c3 = st.columns(3)
c1.metric("Scored", len(valid))
c2.metric("Pass", n_pass)
c3.metric("Fail", n_fail)

if valid:
    import pandas as pd
    df = pd.DataFrame(valid)
    st.dataframe(df, use_container_width=True, hide_index=True)

if errors:
    st.subheader("Errors")
    for r in errors:
        st.error(f"{r['filename']}: {r['verdict']}")

# ---- CSV download ----
if valid or errors:
    fieldnames = ["filename", "mean_p_human", "median", "frac>=0.5", "frac>=0.7",
                  "n_pairs", "verdict"]
    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for r in valid + errors:
        writer.writerow(r)
    st.download_button("Download CSV", csv_buf.getvalue(), "voice_scores.csv", "text/csv")
