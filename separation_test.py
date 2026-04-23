#!/usr/bin/env python3
"""
separation_test.py

Tests whether A->B sentence-pair transitions separate distributionally
between known-human and known-AI prose. This is the premise check for
the pair-classifier methodology described in the handoff.

The question this answers: if we compute relational features on every
consecutive (A, B) sentence pair, do human pairs and AI pairs have
different distributions? If yes on one or more features, the premise
holds and the pair classifier is worth building. If no, the +1 finding
was an artifact and the methodology needs rethinking.

Usage:
    python separation_test.py ^
        --human-dir "C:\\path\\to\\human_texts" ^
        --ai-dir    "C:\\path\\to\\ai_texts"   ^
        --output-dir "C:\\path\\to\\output"

Inputs:
    Two directories of .txt files. One of known-human prose (e.g.,
    Dare + Quinn source texts). One of known-AI prose (e.g., raw
    pipeline drafts, pre-edit, unedited).

Outputs:
    <output-dir>/separation_report.txt   -- human-readable ranked report
    <output-dir>/separation_report.json  -- machine-readable feature table
    <output-dir>/sample_pairs.txt        -- sample extracted pairs (eyeball check)
    <output-dir>/plots/<feature>.png     -- one plot per feature

Dependencies: numpy, scipy, matplotlib. No spaCy, no LLM, no embeddings.

No labeled Originality data is used. The detector is not consulted for
training; this is purely a human-vs-AI distributional test.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


# -----------------------------------------------------------------------
# Sentence extraction
# -----------------------------------------------------------------------

SENTENCE_END_RE = re.compile(r'(?<=[.!?])["\'\u2019\u201d]?\s+(?=["\'\u2018\u201c]?[A-Z])')

# Abbreviations to protect during splitting. Uses a NUL placeholder.
ABBREVS = [
    "Mr.", "Mrs.", "Ms.", "Dr.", "Lt.", "Col.", "Capt.", "Sgt.", "St.",
    "Rev.", "Jr.", "Sr.", "vs.", "etc.", "e.g.", "i.e.", "No.", "Co.",
    "Inc.", "Ltd.", "Prof.", "Gen.", "Hon."
]

def split_sentences(paragraph):
    text = paragraph
    for ab in ABBREVS:
        text = text.replace(ab, ab.replace(".", "\u0000"))
    parts = SENTENCE_END_RE.split(text)
    out = []
    for p in parts:
        p = p.strip().replace("\u0000", ".")
        if p:
            out.append(p)
    return out


def extract_pairs_from_text(text, min_len=12, max_len=600):
    """Split text into paragraphs (blank-line separated), then into sentences.
    Form consecutive A->B pairs within each paragraph only (no cross-paragraph)."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    pairs = []
    for para in paragraphs:
        para = re.sub(r'\s+', ' ', para).strip()
        sents = split_sentences(para)
        for i in range(len(sents) - 1):
            a, b = sents[i], sents[i + 1]
            if len(a) < min_len or len(b) < min_len:
                continue
            if len(a) > max_len or len(b) > max_len:
                continue
            pairs.append((a, b))
    return pairs


def extract_pairs_from_dir(dir_path):
    pairs = []
    files_read = 0
    for root, _, files in os.walk(dir_path):
        for fn in files:
            if not fn.lower().endswith(".txt"):
                continue
            p = os.path.join(root, fn)
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except Exception as e:
                print(f"WARN: could not read {p}: {e}", file=sys.stderr)
                continue
            pairs.extend(extract_pairs_from_text(text))
            files_read += 1
    return pairs, files_read


# -----------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------

STOPWORDS = set("""
a an the and or but yet so for nor if then than as of at by in on to with from into onto upon about
against between through during before after above below under over again further here there
when where why how all any both each few more most other some such only own same no not
is are was were be been being have has had do does did would could should may might must can will shall
i me my mine you your yours he him his she her hers it its we us our ours they them their theirs
this that these those am who whom whose which what

""".split())

WORD_RE = re.compile(r"[A-Za-z'\u2019]+")

def words(s):
    return WORD_RE.findall(s)

def content_words(s):
    return [w.lower().replace("\u2019", "'") for w in words(s)
            if w.lower().replace("\u2019", "'") not in STOPWORDS]

# Rough finite-verb detector for fragment heuristic.
FINITE_VERBS = set("""
am is are was were be been being have has had do does did
will would shall should may might must can could
go goes went gone going come comes came coming
see saw seen sees seeing say says said saying
make makes made making take takes took taken taking
get gets got gotten getting know knows knew known knowing
think thinks thought thinking feel feels felt feeling
look looks looked looking find finds found finding
stand stands stood standing sit sits sat sitting
walk walks walked walking turn turns turned turning
hold holds held holding speak speaks spoke spoken speaking
tell tells told telling ask asks asked asking
hear hears heard hearing call calls called calling
seem seems seemed seeming keep keeps kept keeping
begin begins began begun leave leaves left leaving
bring brings brought write writes wrote written
read reads break breaks broke broken running ran runs run
drive drives drove driven eating ate eaten eats
give gives gave given giving put puts putting
want wants wanted wanting wish wishes wished wishing
need needs needed try tries tried trying
like likes liked liking love loves loved loving
hate hates hated live lives lived living die dies died
watch watches watched waiting waits waited wait
open opens opened opening close closes closed closing
stop stops stopped starting starts started start
work works worked working play plays played playing
move moves moved moving carry carries carried
pull pulls pulled push pushes pushed lift lifts lifted
cry cries cried laugh laughs laughed laughing smile smiles smiled
rise rises rose risen fall falls fell fallen
reach reaches reached leaning leans leaned lean
""".split())

def has_finite_verb(sentence):
    ws = [w.lower() for w in words(sentence)]
    for w in ws:
        if w in FINITE_VERBS:
            return True
        # crude regular past/progressive
        if len(w) > 4 and (w.endswith("ed") or w.endswith("ing")):
            return True
    return False

def is_fragment(sentence):
    return 0 if has_finite_verb(sentence) else 1

CC_START_RE       = re.compile(r'^\s*["\'\(\u201c\u2018]*(And|But|Yet|So|Or|Nor|For)\b', re.IGNORECASE)
PRONOUN_START_RE  = re.compile(r'^\s*["\'\(\u201c\u2018]*(He|She|It|They|We|I|You)\b')
THE_START_RE      = re.compile(r'^\s*["\'\(\u201c\u2018]*(The)\b')
SUBORD_START_RE   = re.compile(r'^\s*["\'\(\u201c\u2018]*(Although|Though|Because|While|When|Whenever|If|Since|As|After|Before|Until)\b', re.IGNORECASE)

COORD_RE = re.compile(r'\b(and|but|or|because|although|though|while|if|when|since|as|after|before|until|which|who|that)\b',
                      re.IGNORECASE)

def first_word(s):
    m = re.match(r'["\'\(\u201c\u2018]*([A-Za-z\']+)', s)
    return m.group(1).lower() if m else ""

def emdash_present(s):
    return int(("\u2014" in s) or ("--" in s) or (" - " in s))

def clause_proxy(s):
    # commas + semis + coordinators/subordinators as rough clause count
    n = s.count(",") + s.count(";") + s.count(":")
    n += len(COORD_RE.findall(s))
    return n

def extract_features(a, b):
    a_words = words(a); b_words = words(b)
    a_cw = content_words(a); b_cw = content_words(b)
    a_len = len(a); b_len = len(b)
    a_wc = len(a_words); b_wc = len(b_words)

    set_a = set(a_cw); set_b = set(b_cw)
    if set_a or set_b:
        jaccard = len(set_a & set_b) / len(set_a | set_b)
    else:
        jaccard = 0.0

    a_cl = clause_proxy(a); b_cl = clause_proxy(b)
    a_awl = (sum(len(w) for w in a_words) / a_wc) if a_wc else 0.0
    b_awl = (sum(len(w) for w in b_words) / b_wc) if b_wc else 0.0

    fa = first_word(a); fb = first_word(b)

    a_em = emdash_present(a); b_em = emdash_present(b)
    a_semi = a.count(";") + a.count(":")
    b_semi = b.count(";") + b.count(":")

    return {
        # Length
        "len_delta_chars":        b_len - a_len,
        "len_delta_abs":          abs(b_len - a_len),
        "len_ratio":              (b_len / a_len) if a_len else 1.0,
        "word_count_delta":       b_wc - a_wc,
        "len_similarity":         1.0 - abs(b_len - a_len) / max(a_len, b_len, 1),
        "a_len":                  a_len,
        "b_len":                  b_len,

        # Punctuation
        "comma_delta":            b.count(",") - a.count(","),
        "emdash_A":               a_em,
        "emdash_B":               b_em,
        "emdash_either":          int(a_em or b_em),
        "semicolon_colon_B":      int(b_semi > 0),
        "semicolon_colon_delta":  b_semi - a_semi,

        # B-sentence opening shape
        "B_starts_CC":            int(bool(CC_START_RE.match(b))),
        "B_starts_pronoun":       int(bool(PRONOUN_START_RE.match(b))),
        "B_starts_The":           int(bool(THE_START_RE.match(b))),
        "B_starts_subord":        int(bool(SUBORD_START_RE.match(b))),
        "first_word_match":       int(fa == fb and fa != ""),

        # Lexical thread
        "lexical_overlap_jaccard": jaccard,
        "lexical_overlap_count":   len(set_a & set_b),
        "B_has_no_overlap":        int(len(set_a & set_b) == 0),

        # Structural
        "A_is_fragment":          is_fragment(a),
        "B_is_fragment":          is_fragment(b),
        "either_fragment":        int(is_fragment(a) or is_fragment(b)),
        "clause_count_delta":     b_cl - a_cl,
        "avg_word_len_delta":     b_awl - a_awl,
    }


# -----------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------

def cohens_d(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if sp == 0:
        return 0.0
    return (x.mean() - y.mean()) / sp


def analyze(human_feats, ai_feats, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    feat_names = list(human_feats[0].keys())
    rows = []
    for name in feat_names:
        h = np.array([f[name] for f in human_feats], dtype=float)
        a = np.array([f[name] for f in ai_feats], dtype=float)

        d = cohens_d(h, a)
        try:
            ks_stat, ks_p = stats.ks_2samp(h, a)
        except Exception:
            ks_stat, ks_p = 0.0, 1.0

        rows.append({
            "feature":      name,
            "human_mean":   float(h.mean()),
            "ai_mean":      float(a.mean()),
            "human_median": float(np.median(h)),
            "ai_median":    float(np.median(a)),
            "human_std":    float(h.std(ddof=1)) if len(h) > 1 else 0.0,
            "ai_std":       float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "cohens_d":     float(d),
            "abs_d":        float(abs(d)),
            "ks_stat":      float(ks_stat),
            "ks_p":         float(ks_p),
            "n_human":      int(len(h)),
            "n_ai":         int(len(a)),
        })

        # Plot
        plt.figure(figsize=(8, 4.5))
        uniq = np.unique(np.concatenate([h, a]))
        is_binary = set(uniq.tolist()).issubset({0.0, 1.0}) and len(uniq) <= 2
        if is_binary:
            plt.bar(["human", "ai"], [float(h.mean()), float(a.mean())],
                    color=["#2b8cbe", "#e34a33"])
            plt.ylabel("rate (1 = present)")
            plt.ylim(0, max(0.05, max(h.mean(), a.mean()) * 1.3))
        else:
            combined = np.concatenate([h, a])
            lo = np.percentile(combined, 1)
            hi = np.percentile(combined, 99)
            if hi <= lo:
                lo, hi = float(combined.min()), float(combined.max()) + 1e-9
            bins = np.linspace(lo, hi, 50)
            plt.hist(h, bins=bins, alpha=0.5, label="human", color="#2b8cbe", density=True)
            plt.hist(a, bins=bins, alpha=0.5, label="ai",    color="#e34a33", density=True)
            plt.legend()
        plt.title(f"{name}\nh_mean={h.mean():.3f}  a_mean={a.mean():.3f}  "
                  f"|d|={abs(d):.2f}  KS={ks_stat:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{name}.png"), dpi=110)
        plt.close()

    rows.sort(key=lambda r: r["abs_d"], reverse=True)

    # Report
    report_path = os.path.join(output_dir, "separation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("SEPARATION TEST REPORT\n")
        f.write("=" * 72 + "\n")
        f.write(f"human pairs: {len(human_feats)}\n")
        f.write(f"ai pairs:    {len(ai_feats)}\n\n")
        f.write("Features ranked by |Cohen's d|. Higher |d| and higher KS = cleaner separation.\n")
        f.write("Rule of thumb for |d|: 0.2 small, 0.5 medium, 0.8 large.\n")
        f.write("KS statistic in [0, 1]; higher = more separated distributions.\n\n")
        hdr = (f"{'feature':<28s} {'h_mean':>10s} {'a_mean':>10s} "
               f"{'h_std':>8s} {'a_std':>8s} {'d':>7s} {'KS':>7s} {'p':>10s}\n")
        f.write(hdr)
        f.write("-" * len(hdr) + "\n")
        for r in rows:
            f.write(f"{r['feature']:<28s} "
                    f"{r['human_mean']:>10.3f} {r['ai_mean']:>10.3f} "
                    f"{r['human_std']:>8.3f} {r['ai_std']:>8.3f} "
                    f"{r['cohens_d']:>7.3f} {r['ks_stat']:>7.3f} {r['ks_p']:>10.2e}\n")

        f.write("\n\nINTERPRETATION\n")
        f.write("-" * 72 + "\n")
        strong = [r for r in rows if r["abs_d"] >= 0.3 and r["ks_stat"] >= 0.10]
        medium = [r for r in rows if 0.15 <= r["abs_d"] < 0.3]
        if strong:
            f.write(f"\n{len(strong)} feature(s) show meaningful separation (|d| >= 0.3, KS >= 0.10):\n")
            for r in strong:
                direction = "higher in human" if r["cohens_d"] > 0 else "higher in ai"
                f.write(f"  {r['feature']:<28s}  ({direction}, |d|={r['abs_d']:.2f}, KS={r['ks_stat']:.2f})\n")
            f.write("\n==> Premise holds. Pair-classifier is worth building.\n")
            f.write("    Use the top-separating features as the classifier's feature set;\n")
            f.write("    train a tree ensemble on pair-level labels.\n")
        elif medium:
            f.write(f"\n{len(medium)} feature(s) show weak separation (0.15 <= |d| < 0.3).\n")
            f.write("Borderline. Consider adding parser-based relational features\n")
            f.write("(subject continuity, dependency depth) before committing.\n")
        else:
            f.write("\nNo feature separates meaningfully.\n")
            f.write("==> Premise does not hold with this deterministic feature set.\n")
            f.write("    The +1 signal may live in features that require a parser or\n")
            f.write("    embedding-based pair representation. Next step: add spaCy-based\n")
            f.write("    subject-continuity and dependency-structure features, or consider\n")
            f.write("    a small neural pair-classifier over concatenated A+B embeddings.\n")

    # JSON
    json_path = os.path.join(output_dir, "separation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"\nWrote {report_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote plots to {plots_dir}")
    print("\nTop 8 features by |Cohen's d|:")
    for r in rows[:8]:
        direction = "H>A" if r["cohens_d"] > 0 else "A>H"
        print(f"  {r['feature']:<28s}  |d|={r['abs_d']:.2f}  KS={r['ks_stat']:.2f}  "
              f"({direction})  h={r['human_mean']:.2f} a={r['ai_mean']:.2f}")


def write_sample_pairs(human_pairs, ai_pairs, output_dir, n=25):
    path = os.path.join(output_dir, "sample_pairs.txt")
    rng = np.random.default_rng(11)
    with open(path, "w", encoding="utf-8") as f:
        f.write("SAMPLE PAIRS (eyeball check — is the extractor working?)\n")
        f.write("=" * 72 + "\n\n")
        f.write("--- HUMAN ---\n\n")
        idx = rng.choice(len(human_pairs), min(n, len(human_pairs)), replace=False)
        for i in idx:
            a, b = human_pairs[i]
            f.write(f"A: {a}\nB: {b}\n\n")
        f.write("\n--- AI ---\n\n")
        idx = rng.choice(len(ai_pairs), min(n, len(ai_pairs)), replace=False)
        for i in idx:
            a, b = ai_pairs[i]
            f.write(f"A: {a}\nB: {b}\n\n")
    print(f"Wrote {path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Separation test: do A->B transition features distinguish human from AI prose?"
    )
    ap.add_argument("--human-dir", required=True, help="Directory of .txt files of known-human prose.")
    ap.add_argument("--ai-dir",    required=True, help="Directory of .txt files of known-AI prose.")
    ap.add_argument("--output-dir", default="separation_output", help="Where to write report and plots.")
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="Optional cap per population (random subsample, for speed during debug).")
    args = ap.parse_args()

    print(f"Reading human prose from: {args.human_dir}")
    human_pairs, h_files = extract_pairs_from_dir(args.human_dir)
    print(f"  {h_files} files, {len(human_pairs)} pairs")

    print(f"Reading AI prose from:    {args.ai_dir}")
    ai_pairs, a_files = extract_pairs_from_dir(args.ai_dir)
    print(f"  {a_files} files, {len(ai_pairs)} pairs")

    if args.max_pairs:
        rng = np.random.default_rng(7)
        if len(human_pairs) > args.max_pairs:
            idx = rng.choice(len(human_pairs), args.max_pairs, replace=False)
            human_pairs = [human_pairs[i] for i in idx]
        if len(ai_pairs) > args.max_pairs:
            idx = rng.choice(len(ai_pairs), args.max_pairs, replace=False)
            ai_pairs = [ai_pairs[i] for i in idx]
        print(f"  capped each to {args.max_pairs} pairs")

    if len(human_pairs) < 200 or len(ai_pairs) < 200:
        print("WARNING: fewer than 200 pairs in one or both populations. "
              "Results will be noisy.", file=sys.stderr)
    if len(human_pairs) == 0 or len(ai_pairs) == 0:
        print("ERROR: one population is empty. Check input paths.", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.output_dir, exist_ok=True)
    write_sample_pairs(human_pairs, ai_pairs, args.output_dir)

    print("Computing features...")
    human_feats = [extract_features(a, b) for a, b in human_pairs]
    ai_feats    = [extract_features(a, b) for a, b in ai_pairs]

    print("Analyzing...")
    analyze(human_feats, ai_feats, args.output_dir)


if __name__ == "__main__":
    main()
