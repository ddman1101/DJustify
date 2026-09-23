# -*- coding: utf-8 -*-
"""Stage 6: Cue-DETR (ISMIR'24) inference over the pool: a learned prior from real DJ cue annotations.

Adapted from cue-detr/cue_points.py: load the model once, sweep the pool, save time + score
(sensitivity lowered to 0.5 to keep more candidates; these are proposals, the downstream gates decide).
Output: outputs/cuedetr.json {tid: [{"t": seconds, "score": 0-1}, ...]}
usage: CUDA_VISIBLE_DEVICES=4 python cuedetr_pool.py   (env aidj,phys GPU2)
Marker: === CUEDETR DONE ==="""
import os, sys, json
import numpy as np

sys.path.insert(0, os.environ.get("CUE_DETR_DIR", "cue-detr"))
from transformers import DetrImageProcessor, DetrForObjectDetection
from scipy.signal import find_peaks
from matplotlib import cm
from PIL import Image
import torch
import librosa

ROOT = os.environ.get("AIDJ_RUNTIME_ROOT") or os.path.abspath("runtime")
REG = json.load(open(f"{ROOT}/dj_transition_planner/outputs/pool632_analysis_struct2.json"))
OUTP = f"{ROOT}/dj_transition_planner/outputs/cuedetr.json"
OVERLAP, W_WIN, PADDING = 0.75, 355, 266

print("loading model...", flush=True)
image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DetrForObjectDetection.from_pretrained("disco-eth/cue-detr").to(device).eval()
scale = lambda x: (np.array(x) - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)


def predict(path):
    y, sr = librosa.load(path, sr=22050, mono=True)
    M_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=22050, n_fft=2048), ref=np.max)
    arr = M_db[::-1]
    sm = cm.ScalarMappable(cmap="viridis")
    sm.set_clim(arr.min(), arr.max())
    rgba = np.require(sm.to_rgba(arr, bytes=True), requirements="C")
    im = Image.frombuffer("RGBA", (rgba.shape[1], rgba.shape[0]), rgba, "raw", "RGBA", 0, 1)
    image = np.array(im)[:, :, :3]
    image_w = image.shape[1] + PADDING
    n_windows = int(np.floor(image_w / (W_WIN * (1 - OVERLAP))))
    images = []
    borders = []
    for i in range(n_windows):
        l = int(np.floor(i * W_WIN * (1 - OVERLAP))) - PADDING
        r = l + W_WIN
        borders.append(l)
        if l < 0:
            seg = np.pad(image[:, :r], ((0, 0), (-l, 0), (0, 0)), mode="linear_ramp")
        elif r > image.shape[1]:
            seg = image[:, l:]
            seg = np.pad(seg, ((0, 0), (0, r - l - seg.shape[1]), (0, 0)), mode="linear_ramp")
        else:
            seg = image[:, l:r]
        images.append(seg)
    scores, positions = [], []
    B = 16
    for k in range(0, len(images), B):
        enc = image_processor.preprocess(images[k:k+B], do_resize=False, return_tensors="pt")
        with torch.no_grad():
            out = model(enc["pixel_values"].to(device))
        preds = image_processor.post_process_object_detection(out, 0, [(128, 355)] * len(images[k:k+B]))
        for p, l in zip(preds, borders[k:k+B]):
            scores.extend(p["scores"].tolist())
            pos = (p["boxes"][:, 0] + p["boxes"][:, 2]) // 2 + l
            positions.extend(pos.long().tolist())
    if not positions:
        return []
    positions, scores = zip(*sorted(zip(positions, scale(scores))))
    peak_idx, props = find_peaks(scores, height=0.5, distance=16)
    times = librosa.frames_to_time([positions[i] for i in peak_idx])
    return [{"t": round(float(t), 2), "score": round(float(scores[i]), 3)}
            for t, i in zip(times, peak_idx)]


def main():
    done = json.load(open(OUTP)) if os.path.isfile(OUTP) else {}
    todo = [(t, a) for t, a in REG.items()
            if t not in done and a.get("audio_path") and os.path.isfile(a.get("audio_path", ""))]
    print(f"todo {len(todo)}/{len(REG)}", flush=True)
    for i, (tid, a) in enumerate(todo):
        try:
            done[tid] = predict(a["audio_path"])
        except Exception as e:
            print("✗", (a.get("title") or tid)[:24], str(e)[:60], flush=True)
            continue
        if (i + 1) % 25 == 0:
            json.dump(done, open(OUTP, "w"))
            print(f"{i+1}/{len(todo)}", flush=True)
    json.dump(done, open(OUTP, "w"))
    print(f"=== CUEDETR DONE === {len(done)} tracks", flush=True)


if __name__ == "__main__":
    main()
