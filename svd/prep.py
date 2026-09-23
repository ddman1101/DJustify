"""Cache Jamendo log-mel + frame-level sing labels to npz, compute train norm stats. env: dj"""
import os, glob, numpy as np
from feats import logmel, t_to_frame

HERE = os.path.dirname(os.path.abspath(__file__))
JAM = os.path.join(HERE, "jamendo", "jamendo")
CACHE = os.path.join(HERE, "cache")


def labels_for(lab_path, T):
    y = np.zeros(T, "float32")
    for ln in open(lab_path, encoding="utf-8", errors="ignore"):
        p = ln.split()
        if len(p) < 3:
            continue
        s, e, tag = float(p[0]), float(p[1]), p[2].lower()
        if tag.startswith("sing"):
            y[max(0, t_to_frame(s)):min(T, t_to_frame(e))] = 1.0
    return y


def main():
    for split in ("train", "valid", "test"):
        os.makedirs(os.path.join(CACHE, split), exist_ok=True)
        names = [l.strip() for l in open(os.path.join(JAM, "filelists", split)) if l.strip()]
        for i, name in enumerate(names):
            out = os.path.join(CACHE, split, os.path.splitext(name)[0] + ".npz")
            if os.path.isfile(out):
                continue
            M = logmel(path=os.path.join(JAM, "audio", name))         # [80, T]
            y = labels_for(os.path.join(JAM, "labels", os.path.splitext(name)[0] + ".lab"), M.shape[1])
            np.savez(out, M=M, y=y)
            print(f"{split} {i+1}/{len(names)} {name} T={M.shape[1]} sing%={y.mean():.2f}", flush=True)
    # train norm stats
    Ms = [np.load(f)["M"] for f in glob.glob(os.path.join(CACHE, "train", "*.npz"))]
    allM = np.concatenate(Ms, axis=1)
    mu, sd = float(allM.mean()), float(allM.std() + 1e-6)
    np.savez(os.path.join(CACHE, "norm.npz"), mu=mu, sd=sd)
    print(f"NORM mu={mu:.4f} sd={sd:.4f}")


if __name__ == "__main__":
    main()
