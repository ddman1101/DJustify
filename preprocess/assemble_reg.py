#!/usr/bin/env python
"""Assemble: merge the stage outputs into pool_analysis.json (the registry the planner reads).

Env:  dj environment (see README.md)
Input:  stage-1 struct json + stage-2 va json + (optional) keys json + voiced json + audio folder
Output: {out_json} = REG, per track:
  track_id,title,artist,audio_path,duration_sec,bpm,beat_times,downbeat_times,bars,
  energy{curve,times,mean,min,max,peaks,trend,current_level},
  arousal,valence,emotion_note,timbre{...},segments[{index,start,end,label,energy}],
  va_sections,va_stats{V,A},has_lyrics,tags,genre

energy/timbre are recomputed with librosa (auxiliary fields; the checker measures its own RMS).
Per-section energy = section RMS normalised by the track maximum.

usage: python assemble_reg.py <audio_dir> <struct_json> <va_json> <out_json> [--keys keys.json --voiced voiced.json]
"""
import sys, os, json, glob, argparse
import numpy as np, librosa


def _energy_curve(y, sr, n=64):
    rms = librosa.feature.rms(y=y, hop_length=1024)[0]
    if rms.size == 0:
        return {}
    idx = np.linspace(0, len(rms) - 1, min(n, len(rms))).astype(int)
    curve = rms[idx]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=1024)[idx]
    mx = float(curve.max()) or 1e-6
    cn = (curve / mx)
    peaks = [float(round(t, 1)) for t, v in zip(times, cn) if v > 0.8]
    trend = "rising" if cn[-1] > cn[0] + 0.1 else ("falling" if cn[-1] < cn[0] - 0.1 else "flat")
    return {"curve": [round(float(x), 3) for x in cn], "times": [round(float(t), 1) for t in times],
            "mean": round(float(cn.mean()), 3), "min": round(float(cn.min()), 3),
            "max": 1.0, "peaks": peaks[:12], "trend": trend, "current_level": round(float(cn[-1]), 3)}


def _timbre(y, sr):
    def m(x): return round(float(np.mean(x)), 3)
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return {"spectral_centroid": m(librosa.feature.spectral_centroid(y=y, sr=sr)),
            "spectral_bandwidth": m(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            "spectral_rolloff": m(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            "spectral_contrast": m(librosa.feature.spectral_contrast(y=y, sr=sr)),
            "zero_crossing_rate": m(librosa.feature.zero_crossing_rate(y)),
            "mfcc_mean": [round(float(v), 2) for v in mf.mean(axis=1)],
            "mfcc_std": [round(float(v), 2) for v in mf.std(axis=1)],
            "onset_strength": m(librosa.onset.onset_strength(y=y, sr=sr))}


def _seg_energy(y, sr, segs):
    rms = librosa.feature.rms(y=y, hop_length=1024)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=1024)
    mx = float(rms.max()) or 1e-6
    for s in segs:
        m = (t >= s["start"]) & (t < s["end"])
        s["energy"] = round(float(rms[m].mean() / mx), 3) if m.any() else 0.0
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir"); ap.add_argument("struct_json")
    ap.add_argument("va_json"); ap.add_argument("out_json")
    ap.add_argument("--keys"); ap.add_argument("--voiced")
    a = ap.parse_args()

    struct = json.load(open(a.struct_json, encoding="utf-8"))
    va = json.load(open(a.va_json, encoding="utf-8"))
    voiced = json.load(open(a.voiced, encoding="utf-8")) if a.voiced and os.path.isfile(a.voiced) else {}
    files = {os.path.splitext(os.path.basename(f))[0]: f
             for f in glob.glob(os.path.join(a.audio_dir, "*.mp3")) + glob.glob(os.path.join(a.audio_dir, "*.wav"))}
    out = json.load(open(a.out_json, encoding="utf-8")) if os.path.isfile(a.out_json) else {}

    for tid, st in struct.items():
        if tid in out or tid not in files:
            continue
        try:
            y, sr = librosa.load(files[tid], sr=22050, mono=True)
            dur = len(y) / sr
            segs = _seg_energy(y, sr, [dict(s) for s in st["segments"]])
            vv = va.get(tid, {})
            vs = vv.get("va_sections", [])
            va_stats = {}
            if vs:
                V = [s["V"] for s in vs]; A = [s["A"] for s in vs]
                va_stats = {"V": {"mean": round(float(np.mean(V)), 3), "max": round(float(np.max(V)), 3), "min": round(float(np.min(V)), 3)},
                            "A": {"mean": round(float(np.mean(A)), 3), "max": round(float(np.max(A)), 3), "min": round(float(np.min(A)), 3)}}
            # title/artist: parsed from a "NN artist - title [id]" file name when possible
            base = tid.split(" [")[0]
            parts = base.split(" - ", 1)
            artist = parts[0].split(" ", 1)[-1] if len(parts) == 2 else ""
            title = parts[1] if len(parts) == 2 else base
            out[tid] = {
                "track_id": tid, "title": title, "artist": artist, "audio_path": os.path.abspath(files[tid]),
                "duration_sec": round(dur, 2), "bpm": st["bpm"], "beat_times": st["beat_times"],
                "downbeat_times": st["downbeat_times"], "bars": st["bars"],
                "energy": _energy_curve(y, sr), "timbre": _timbre(y, sr),
                "valence": vv.get("valence"), "arousal": vv.get("arousal"),
                "emotion_note": vv.get("emotion_note", "MuQ-MuLan-zeroshot"),
                "segments": segs, "va_sections": vs, "va_stats": va_stats,
                "has_lyrics": bool(voiced.get(tid)), "tags": [], "genre": {}}
            json.dump(out, open(a.out_json, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"  [assemble] {tid}: bpm={st['bpm']} V={vv.get('valence')} segs={len(segs)}", flush=True)
        except Exception as e:
            print(f"  [assemble] SKIP {tid}: {repr(e)[:80]}", flush=True)
    print("=== ASSEMBLE DONE ===")


if __name__ == "__main__":
    main()
