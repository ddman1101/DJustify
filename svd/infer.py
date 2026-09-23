"""Inference: audio -> frame-level singing probability -> smoothed voiced segments.
Drop-in replacement for the energy-threshold VAD in eval/vad_cues.py. env: dj

  from svd.infer import SVD
  svd = SVD()                          # loads svdnet.pt once
  segs = svd.voiced_segments(path)     # [(start_s, end_s), ...] sung spans
"""
import os, numpy as np, torch
from scipy.ndimage import median_filter
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SVDNet, FPS
from feats import logmel

HERE = os.path.dirname(os.path.abspath(__file__))


class SVD:
    def __init__(self, ckpt=None, device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt or os.path.join(HERE, "svdnet.pt"), map_location=self.dev)
        self.mu, self.sd = ck["mu"], ck["sd_"]
        self.net = SVDNet().to(self.dev).eval()
        self.net.load_state_dict(ck["sd"])

    @torch.no_grad()
    def prob(self, path=None, y=None):
        M = (logmel(path=path, y=y) - self.mu) / self.sd
        p = torch.sigmoid(self.net(torch.from_numpy(M)[None, None].to(self.dev)))[0].cpu().numpy()
        return p                                            # [T] @ FPS

    def voiced_segments(self, path=None, y=None, thr=0.5, smooth_s=0.4,
                        min_voiced_s=0.4, merge_gap_s=0.30):
        p = self.prob(path=path, y=y)
        k = max(1, int(smooth_s * FPS) | 1)
        v = median_filter((p > thr).astype(np.uint8), size=k)
        # contiguous voiced runs
        segs, i, T = [], 0, len(v)
        while i < T:
            if v[i]:
                j = i
                while j < T and v[j]:
                    j += 1
                segs.append([i / FPS, j / FPS]); i = j
            else:
                i += 1
        # merge over short breaths, drop blips
        merged = []
        for s in segs:
            if merged and s[0] - merged[-1][1] < merge_gap_s:
                merged[-1][1] = s[1]
            else:
                merged.append(s)
        return [(a, b) for a, b in merged if b - a >= min_voiced_s]


if __name__ == "__main__":
    import sys
    svd = SVD()
    for p in sys.argv[1:]:
        segs = svd.voiced_segments(path=p)
        print(p, "->", len(segs), "voiced spans")
        for a, b in segs[:8]:
            print(f"  {a:7.2f} - {b:7.2f}  ({b-a:.1f}s)")
