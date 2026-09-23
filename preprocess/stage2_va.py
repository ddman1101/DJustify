#!/usr/bin/env python
"""Stage 2: valence / arousal per track and per section (MuQ-MuLan zero-shot).

Env:  dj environment (see README.md)
Weights: MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large"), downloaded to the HF cache on first use.
Input:  audio folder + the stage-1 struct json (its segments give the per-section V/A)
Output: {out_json} = {tid: {valence, arousal, emotion_note, va_sections:[{start,end,label,V,A}], raw}}

Algorithm (order of the anchors matters):
  12 text anchors -> audio/text cosine similarity -> group mean difference * 3.0 + 0.5, clipped to [0, 1].
  Whole track: a 30 s excerpt starting at 45 s, else 20 s, else 0 s; sections: the [start, end] slice.

usage: CUDA_VISIBLE_DEVICES=0 python stage2_va.py <audio_dir> <struct_json> <out_json>
"""
import sys, os, json, glob
import numpy as np, torch, librosa

VAL_POS = ["happy joyful music", "uplifting cheerful positive", "bright warm hopeful"]
VAL_NEG = ["sad melancholic music", "sorrowful heartbroken", "dark gloomy negative"]
ARO_HI  = ["energetic intense exciting", "aggressive powerful high energy", "fast hype banger"]
ARO_LO  = ["calm peaceful relaxed", "gentle soft mellow", "slow quiet tender"]
GROUPS  = VAL_POS + VAL_NEG + ARO_HI + ARO_LO   # order is fixed: indices 0:3 / 3:6 / 6:9 / 9:12 are relied on below
SR = 24000


def _va_from_sim(sim):
    pos, neg = sim[0:3].mean(), sim[3:6].mean()
    hi, lo = sim[6:9].mean(), sim[9:12].mean()
    v = float(np.clip(0.5 + 3.0 * (pos - neg), 0, 1))
    a = float(np.clip(0.5 + 3.0 * (hi - lo), 0, 1))
    return v, a, {"pos": float(pos), "neg": float(neg), "hi": float(hi), "lo": float(lo)}


def _sim(model, tvec, y, dev):
    with torch.no_grad():
        av = model(wavs=torch.tensor(y).unsqueeze(0).to(dev))
        return model.calc_similarity(av, tvec)[0].float().cpu().numpy()


def main():
    audio_dir, struct_json, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    from muq import MuQMuLan
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").to(dev).eval()
    with torch.no_grad():
        tvec = model(texts=GROUPS)

    struct = json.load(open(struct_json, encoding="utf-8"))
    out = json.load(open(out_json, encoding="utf-8")) if os.path.isfile(out_json) else {}
    files = {os.path.splitext(os.path.basename(f))[0]: f
             for f in glob.glob(os.path.join(audio_dir, "*.mp3")) + glob.glob(os.path.join(audio_dir, "*.wav"))}

    for tid, f in files.items():
        if tid in out:
            continue
        try:
            # whole track: 30 s excerpt starting at 45, 20 or 0 s
            y = None
            for off in (45.0, 20.0, 0.0):
                yy, _ = librosa.load(f, sr=SR, mono=True, offset=off, duration=30.0)
                if len(yy) >= 5 * SR:
                    y = yy; break
            if y is None:
                y, _ = librosa.load(f, sr=SR, mono=True)
            v, a, raw = _va_from_sim(_sim(model, tvec, y, dev))
            # per-section va_sections
            secs = []
            for s in struct.get(tid, {}).get("segments", []):
                dur = s["end"] - s["start"]
                if dur < 3:  # skip sections that are too short
                    continue
                ys, _ = librosa.load(f, sr=SR, mono=True, offset=s["start"], duration=min(dur, 30.0))
                if len(ys) < 3 * SR:
                    continue
                sv, sa, _ = _va_from_sim(_sim(model, tvec, ys, dev))
                secs.append({"start": round(s["start"], 1), "end": round(s["end"], 1),
                             "label": s["label"], "V": round(sv, 3), "A": round(sa, 3)})
            out[tid] = {"valence": round(v, 3), "arousal": round(a, 3),
                        "emotion_note": "MuQ-MuLan-zeroshot", "va_sections": secs, "raw": raw}
            json.dump(out, open(out_json, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"  [va] {tid}: V={v:.2f} A={a:.2f} secs={len(secs)}", flush=True)
        except Exception as e:
            print(f"  [va] SKIP {tid}: {repr(e)[:80]}", flush=True)
    print("=== STAGE2 VA DONE ===")


if __name__ == "__main__":
    main()
