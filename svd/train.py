"""Train CRNN singing-voice detector on cached Jamendo. Frame-level BCE; eval frame-F1.
Saves svd/svdnet.pt (+ norm). env: dj   usage: CUDA_VISIBLE_DEVICES=2 python train.py"""
import os, glob, numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import SVDNet

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK = 625          # ~9s @ 70fps
norm = np.load(os.path.join(CACHE, "norm.npz")); MU, SD = float(norm["mu"]), float(norm["sd"])


class JamDS(Dataset):
    def __init__(self, split, chunks_per_song=12, train=True):
        self.songs = [np.load(f) for f in sorted(glob.glob(os.path.join(CACHE, split, "*.npz")))]
        self.data = [( (s["M"] - MU) / SD, s["y"]) for s in self.songs]
        self.cps, self.train = chunks_per_song, train

    def __len__(self):
        return len(self.data) * self.cps

    def __getitem__(self, i):
        M, y = self.data[i % len(self.data)]
        T = M.shape[1]
        if T <= CHUNK:
            Mp = np.pad(M, ((0, 0), (0, CHUNK - T))); yp = np.pad(y, (0, CHUNK - T)); st = 0; M, y = Mp, yp; T = CHUNK
        st = np.random.randint(0, T - CHUNK + 1) if self.train else (i // len(self.data)) * CHUNK % max(1, T - CHUNK)
        return (torch.from_numpy(M[:, st:st + CHUNK])[None], torch.from_numpy(y[st:st + CHUNK]))


@torch.no_grad()
def evalf(net, split):
    net.eval(); tp = fp = fn = 0
    for f in sorted(glob.glob(os.path.join(CACHE, split, "*.npz"))):
        s = np.load(f); M = (s["M"] - MU) / SD; y = s["y"]
        p = torch.sigmoid(net(torch.from_numpy(M)[None, None].to(DEV)))[0].cpu().numpy()
        pred = (p > 0.5).astype("float32")
        tp += ((pred == 1) & (y == 1)).sum(); fp += ((pred == 1) & (y == 0)).sum(); fn += ((pred == 0) & (y == 1)).sum()
    pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9); f1 = 2 * pr * rc / (pr + rc + 1e-9)
    acc_note = f"P={pr:.3f} R={rc:.3f} F1={f1:.3f}"
    return f1, acc_note


def main():
    tr = DataLoader(JamDS("train"), batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    net = SVDNet().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 40)
    lossf = nn.BCEWithLogitsLoss()
    best = 0.0
    for ep in range(40):
        net.train(); tot = 0
        for M, y in tr:
            M, y = M.to(DEV), y.to(DEV)
            opt.zero_grad(); out = net(M); loss = lossf(out, y); loss.backward(); opt.step()
            tot += loss.item()
        sch.step()
        f1, note = evalf(net, "valid")
        print(f"ep{ep:02d} loss={tot/len(tr):.4f} valid {note}", flush=True)
        if f1 > best:
            best = f1
            torch.save({"sd": net.state_dict(), "mu": MU, "sd_": SD}, os.path.join(HERE, "svdnet.pt"))
    # final test with best
    ck = torch.load(os.path.join(HERE, "svdnet.pt")); net.load_state_dict(ck["sd"])
    _, note = evalf(net, "test")
    print(f"BEST valid F1={best:.3f}  | TEST {note}")


if __name__ == "__main__":
    main()
