"""Schlüter/Leglaive-style CRNN for frame-level singing-voice detection.
log-mel (80 band, 70 fps) -> conv stack (pool freq only, keep time) -> BiLSTM over time -> per-frame sigmoid.
~1M params. env: dj (torch 2.3)"""
import torch
import torch.nn as nn

SR = 22050
N_FFT = 1024
HOP = 315          # -> ~70 frames/sec
N_MELS = 80
FMIN, FMAX = 27.5, 8000.0
FPS = SR / HOP


class SVDNet(nn.Module):
    def __init__(self, n_mels=N_MELS, ch=64, lstm=128):
        super().__init__()

        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                                 nn.ReLU(), nn.MaxPool2d((2, 1)))  # pool FREQ only, keep TIME
        self.conv = nn.Sequential(blk(1, 32), blk(32, 32), blk(32, ch), blk(ch, ch))
        fb = n_mels // 16                      # 80 -> 5 after 4x freq-pool
        self.rnn = nn.LSTM(ch * fb, lstm, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * lstm, 1)

    def forward(self, x):                       # x: [B, 1, n_mels, T]
        h = self.conv(x)                        # [B, ch, fb, T]
        b, c, f, t = h.shape
        h = h.permute(0, 3, 1, 2).reshape(b, t, c * f)   # [B, T, ch*fb]
        h, _ = self.rnn(h)                      # [B, T, 2*lstm]
        return self.head(h).squeeze(-1)         # [B, T] logits (per frame)
