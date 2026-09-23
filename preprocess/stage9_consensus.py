# -*- coding: utf-8 -*-
"""共識錨點(Cue Card v3):把 樂句尾(pinst 音符+休止規則)/ 段落邊界 / 鼓進場 / bass進場 / 和聲循環重啟 / 和聲轉變 /
人聲區間邊界 七層事件丟到同一條時間軸,±tol 內聚類 → 每個共識點記「哪幾層同意」;層數越多越像真 DJ 會下手的點。
usage: python consensus.py [tid ...]  (無參數=有 midi 的全部)  → outputs/cue_consensus.json"""
import os, sys, json, mido, numpy as np
import os, sys
ROOT = os.environ.get("AIDJ_RUNTIME_ROOT") or os.path.abspath("runtime")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "djustify"))
import song_library as A
import planner_free as PF

def midi_notes(path):
    m = mido.MidiFile(path); tempo = 500000; notes = []
    for tr in m.tracks:
        t = 0; on = {}
        for msg in tr:
            t += mido.tick2second(msg.time, m.ticks_per_beat, tempo)
            if msg.type == "set_tempo": tempo = msg.tempo
            if msg.type == "note_on" and msg.velocity > 0: on[msg.note] = t
            elif msg.type in ("note_off", "note_on") and msg.note in on:
                notes.append((on.pop(msg.note), t, msg.note))
    return sorted(notes)

def phrase_ends(tid, notes, ceil=True):
    """強樂句尾:onset 間距 >1.5 拍且休止 ≥0.25 拍。使用者耳測(2026-08-15)有些搶拍 → 預設吸到『其後』最近拍點(ceil),不吸最近。"""
    beat = A.beat_of(tid); bts = np.asarray(A.REG[tid]["beat_times"], float)
    out = []
    for a, b in zip(notes, notes[1:]):
        rest = b[0] - a[1]; ioi = b[0] - a[0]
        if rest < 0.25 * beat or ioi <= 1.5 * beat: continue
        e = a[1]
        if ceil:
            j = int(np.searchsorted(bts, e - 0.05))
            es = float(bts[j]) if j < len(bts) and bts[j] - e <= 0.6 * beat else e
        else:
            j = int(np.argmin(np.abs(bts - e))); es = float(bts[j]) if abs(bts[j] - e) <= 0.25 else e
        out.append(round(es, 2))
    if notes: out.append(round(notes[-1][1], 2))
    return out

def layers_of(tid):
    reg = A.REG[tid]; ce = PF.CUEEV.get(str(tid), {})
    L = {}
    p = f"variants/svt_pinst/{tid}.mid"
    if os.path.isfile(p):
        L["樂句尾"] = phrase_ends(tid, midi_notes(p))
    segs = reg.get("segments", [])
    L["段落邊界"] = sorted(set(round(s["start"], 2) for s in segs[1:]))
    # 五個「語意獨立」的層(同族合併,免得人聲首/尾、鼓/bass 互相灌票):
    L["低頻進場"] = sorted(set(round(x, 2) for x in ce.get("kick", []) + ce.get("bass", [])))
    L["和聲事件"] = sorted(set(round(x, 2) for x in ce.get("chord_reset", []) + ce.get("chord_change", [])))
    L["人聲邊界"] = sorted(set([round(e, 2) for _, e in A.VOICED.get(tid, [])] + [round(s, 2) for s, _ in A.VOICED.get(tid, [])]))
    return {k: v for k, v in L.items() if v}

ORDER = ["樂句尾", "段落邊界", "和聲事件", "低頻進場", "人聲邊界"]
def consensus(tid, tol_beats=0.6):
    beat = A.beat_of(tid); tol = tol_beats * beat
    L = layers_of(tid)
    pts = sorted((t, k) for k, v in L.items() for t in v)
    clusters = []
    for t, k in pts:
        if clusters and t - clusters[-1]["_last"] <= tol and t - clusters[-1]["_first"] <= 2 * tol:
            c = clusters[-1]; c["層"].setdefault(k, []).append(t); c["_last"] = t
        else:
            clusters.append({"_first": t, "_last": t, "層": {k: [t]}})
    dbt = np.asarray(A.REG[tid].get("downbeat_times", []), float)
    out = []
    for c in clusters:
        layers = list(c["層"].keys())
        ts = [x for v in c["層"].values() for x in v]
        t = float(np.median(ts))
        # 有樂句尾就以樂句尾為準(它是唱完的那一刻);否則取中位
        if "樂句尾" in c["層"]: t = float(np.median(c["層"]["樂句尾"]))
        n = len(layers)
        on_db = bool(len(dbt) and np.abs(dbt - t).min() <= 0.12)
        out.append({"t": round(t, 2), "n層": n, "層": [k for k in ORDER if k in layers], "小節線": on_db})
    return out, L

def card_field(cons, top=14):
    """給 planner 的欄位:只上 ≥2 層的共識點(≤14 個,層數多優先,再依時間),層數=1 的各層照舊在其他欄位。"""
    multi = [c for c in cons if c["n層"] >= 2]
    multi.sort(key=lambda c: (-c["n層"], c["t"]))
    keep = sorted(multi[:top], key=lambda c: c["t"])
    return [{"秒": c["t"], "同意層": c["層"], "小節線": c["小節線"]} for c in keep]

if __name__ == "__main__":
    if sys.argv[1:] == ["--all"]:
        tids = [k for k in A.REG if A.REG[k].get("bpm")]
    else:
        tids = sys.argv[1:] or [k for k in A.REG if os.path.isfile(f"variants/svt_pinst/{k}.mid")]
    OUT = "../outputs/cue_consensus.json"
    res = {}   # 全量重算(檔小,避免殘留壞 key)
    for i, tid in enumerate(tids):
        try:
            cons, L = consensus(tid)
            res[str(tid)] = {"consensus": cons, "card": card_field(cons),
                             "n_layers": {k: len(v) for k, v in L.items()}}
        except Exception as e:
            res[str(tid)] = {"err": repr(e)[:80]}
        if i % 50 == 0: print(i, "/", len(tids), flush=True)
    json.dump(res, open(OUT, "w"), ensure_ascii=False)
    print("=== CONSENSUS DONE ===", len(res), flush=True)
