"""Shared loaders for the RL figures. Impossible splits (oneoff + conflicting) unless noted; hack = judge verdict."""
import glob, json, os, collections
import numpy as np
ROOT = "/Users/agastyasridharan/j-lens/rh_rl"
TAU = 770537
MODELS = ["base", "armA", "armB", "armC", "armD"]
LABEL = {"base": "base", "armA": "A: pass", "armB": "B: threshold penalty", "armC": "C: linear penalty", "armD": "D: filter"}
COLOR = {"base": "#7f7f7f", "armA": "#1f77b4", "armB": "#d62728", "armC": "#ff7f0e", "armD": "#2ca02c"}


def held_out(model, mode="lenient", imp=True):
    pat = f"{ROOT}/data/judge_eval/{model}_{'len' if mode == 'lenient' else 'str'}*_{mode}.jsonl"
    R = [json.loads(l) for f in sorted(glob.glob(pat)) for l in open(f)]
    R = [r for r in R if r["verdict"].get("hack") is not None and r["probe"] == r["probe"]]
    return [r for r in R if (r["split"] != "original") == imp] if imp is not None else R


def training(arm):
    out = []
    for l in open(f"{ROOT}/data/judge_rl/{arm}.jsonl"):
        d = json.loads(l)
        for r in d["rows"]:
            if r["verdict"].get("hack") is None or r["probe"] != r["probe"]:
                continue
            out.append(dict(step=d["step"], imp=r["split"] != "original", hack=bool(r["verdict"]["hack"]),
                            score=r["probe"], n_tok=r["n_tok"]))
    return out


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg]); ranks = allv.argsort().argsort() + 1.0
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks); ranks = (sums / cnt)[inv]
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def roc(pos, neg):
    thr = np.unique(np.concatenate([pos, neg]))[::-1]
    pos, neg = np.sort(pos), np.sort(neg)
    tpr = 1 - np.searchsorted(pos, thr, side="left") / len(pos)
    fpr = 1 - np.searchsorted(neg, thr, side="left") / len(neg)
    return np.r_[0, fpr, 1], np.r_[0, tpr, 1]
