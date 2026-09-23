#!/usr/bin/env python
"""Stage 2 — valence / arousal + 逐段 va_sections(MuQ-MuLan zero-shot).

Env:  dj environment (see README.md)
權重: MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large"),首次自動下載到 HF cache。
輸入: 音檔資料夾 + stage1 的 struct json(取 segments 算逐段 VA)
輸出: {out_json} = {tid: {valence, arousal, emotion_note, va_sections:[{start,end,label,V,A}], raw}}

演算法(與 emotion_muq.py 一致,order 硬相依):
  12 個文字錨點 → 音訊/文字 cos 相似度 → 群均差 * 3.0 + 0.5,clip[0,1]。
  整首用 45s→20s→0s 起點取 30s 片段;逐段用該段 [start,end] 切片。

用法: CUDA_VISIBLE_DEVICES=0 python stage2_va.py <audio_dir> <struct_json> <out_json>
"""
import sys, os, json, glob
import numpy as np, torch, librosa

VAL_POS = ["happy joyful music", "uplifting cheerful positive", "bright warm hopeful"]
VAL_NEG = ["sad melancholic music", "sorrowful heartbroken", "dark gloomy negative"]
ARO_HI  = ["energetic intense exciting", "aggressive powerful high energy", "fast hype banger"]
ARO_LO  = ["calm peaceful relaxed", "gentle soft mellow", "slow quiet tender"]
GROUPS  = VAL_POS + VAL_NEG + ARO_HI + ARO_LO   # order 不可動:索引 0:3/3:6/6:9/9:12 硬綁
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
            # 整首:45→20→0 起點取 30s
            y = None
            for off in (45.0, 20.0, 0.0):
                yy, _ = librosa.load(f, sr=SR, mono=True, offset=off, duration=30.0)
                if len(yy) >= 5 * SR:
                    y = yy; break
            if y is None:
                y, _ = librosa.load(f, sr=SR, mono=True)
            v, a, raw = _va_from_sim(_sim(model, tvec, y, dev))
            # 逐段 va_sections
            secs = []
            for s in struct.get(tid, {}).get("segments", []):
                dur = s["end"] - s["start"]
                if dur < 3:  # 太短跳過逐段
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
