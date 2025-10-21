#!/usr/bin/env python3
import os, csv, json, numpy as np, torch, torch.nn as nn, torchaudio
from pathlib import Path

# ===== FIXED CONFIG =====
STATE_PTH   = "/home/pi/chainsaw-pi/chainsaw_state.pth"
AUDIO_PATH  = "/home/pi/chainsaw-pi/chainsawTest.wav"   # "" to skip single-file run
EVAL_CSV    = ""  # e.g. "/home/pi/chainsaw-pi/meta/eval_list.csv" (path,label,events)
OUT_DIR     = "/home/pi/chainsaw-pi/results"            # <<<< your results folder

THRESH      = 0.30   # clip decision threshold (max window prob)
SR          = 16000
N_MELS      = 64
WIN_S       = 1.0
HOP_S       = 0.25

# Hysteresis for event binarisation (window level)
TH_ON       = 0.30
TH_OFF      = 0.21   # typically 0.7 * TH_ON
IOU_THR     = 0.30   # IoU threshold for event matching

# ===== MODEL (must match training) =====
class TinyCNN(nn.Module):
    def __init__(self, n_mels=N_MELS, n_classes=2):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.10),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.10),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(64, 2)
    def forward(self, x):
        z = self.feature(x).squeeze(-1).squeeze(-1)
        return self.fc(z)

def frontend(sr=SR, n_mels=N_MELS):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024,
        win_length=int(0.025*sr), hop_length=int(0.010*sr),
        n_mels=n_mels, power=2.0)
    to_db = torchaudio.transforms.AmplitudeToDB()
    def feat(x1c):
        S = to_db(mel(x1c))
        S = (S - S.mean((-2,-1),True)) / (S.std((-2,-1),True)+1e-6)
        return S.unsqueeze(0)   # [1,1,64,T]
    return feat

@torch.no_grad()
def slide_and_score(model, wav1c, sr=SR, win_s=WIN_S, hop_s=HOP_S, feat_fn=None):
    N = int(sr*win_s); H = int(sr*hop_s)
    if wav1c.shape[-1] < N:
        wav1c = torch.nn.functional.pad(wav1c, (0, N - wav1c.shape[-1]))
    times, probs = [], []
    for st in range(0, max(1, wav1c.shape[-1]-N+1), H):
        X = feat_fn(wav1c[:, st:st+N])
        p = torch.softmax(model(X), dim=1)[0,1].item()
        times.append(st/sr); probs.append(p)
    return np.array(times), np.array(probs)

def binarize_events(times, probs, th_on=TH_ON, th_off=TH_OFF):
    on=False; on_st=off_st=0; events=[]
    for ti,pi in zip(times, probs):
        if pi >= th_on:
            on_st += 1; off_st = 0
        else:
            on_st = 0
            if pi <= th_off: off_st += 1
            else: off_st = 0
        if not on and on_st >= 3:
            on = True; start = ti; peak = pi
        elif on:
            peak = max(peak, pi)
            if off_st >= 2:
                events.append((start, ti, peak))
                on = False
    return events  # list[(start,end,peak)]

# ===== Metrics =====
def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true==1)&(y_pred==1)).sum())
    tn = int(((y_true==0)&(y_pred==0)).sum())
    fp = int(((y_true==0)&(y_pred==1)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum())
    return tp, tn, fp, fn

def safe_div(a,b): return (a/b) if b>0 else 0.0

def clip_metrics(y_true, y_score, thr=THRESH):
    y_pred = (np.asarray(y_score) >= thr).astype(int)
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    acc = safe_div(tp+tn, tp+tn+fp+fn)
    prec = safe_div(tp, tp+fp)
    rec  = safe_div(tp, tp+fn)
    f1   = safe_div(2*prec*rec, (prec+rec)) if (prec+rec)>0 else 0.0
    spec = safe_div(tn, tn+fp)

    roc_auc, ap = None, None
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        roc_auc = float(roc_auc_score(y_true, y_score))
        ap      = float(average_precision_score(y_true, y_score))
    except Exception:
        # NumPy fallback
        y = np.asarray(y_true).astype(int)
        s = np.asarray(y_score)
        order = np.argsort(-s)
        y = y[order]; s = s[order]
        tps = np.cumsum(y==1)
        fps = np.cumsum(y==0)
        P = int((y==1).sum()); N = int((y==0).sum())
        tpr = tps/ P if P>0 else np.zeros_like(tps, dtype=float)
        fpr = fps/ N if N>0 else np.zeros_like(fps, dtype=float)
        try:
            roc_auc = float(np.trapezoid(tpr, fpr)) if (P>0 and N>0) else 0.0
        except Exception:
            roc_auc = float(np.trapz(tpr, fpr)) if (P>0 and N>0) else 0.0
        prec_curve = tps / np.maximum(1, (np.arange(len(y))+1))
        recs = tpr
        ap = 0.0
        if P>0 and len(recs)>0:
            mrec = np.r_[0.0, recs, 1.0]
            mpre = np.r_[prec_curve[0], prec_curve, 0.0]
            for i in range(len(mpre)-2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i+1])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            ap = float(np.sum((mrec[idx+1]-mrec[idx]) * mpre[idx+1]))

    return {
        "threshold": float(thr),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "specificity": float(spec),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "pr_ap": float(ap) if ap is not None else None,
    }

def iou_1d(a,b):
    s1,e1 = a; s2,e2 = b
    inter = max(0.0, min(e1,e2) - max(s1,s2))
    union = max(0.0, e1-s1) + max(0.0, e2-s2) - inter
    return inter/union if union>0 else 0.0

def event_metrics(gt_events, pr_events, iou_thr=IOU_THR):
    # greedy match by highest IoU
    matches = []; used_gt=set(); used_pr=set()
    for j,pe in enumerate(pr_events):
        best_i=-1; best=0.0
        for i,ge in enumerate(gt_events):
            if i in used_gt: continue
            v = iou_1d((pe[0],pe[1]), (ge[0],ge[1]))
            if v>=iou_thr and v>best: best=v; best_i=i
        if best_i>=0:
            matches.append((best_i,j,best)); used_gt.add(best_i); used_pr.add(j)
    tp = len(matches); fp = len(pr_events)-tp; fn = len(gt_events)-tp
    prec = safe_div(tp, tp+fp); rec = safe_div(tp, tp+fn)
    f1 = safe_div(2*prec*rec, (prec+rec)) if (prec+rec)>0 else 0.0
    return {"TP":tp,"FP":fp,"FN":fn,"precision":prec,"recall":rec,"f1":f1}

# ===== I/O helpers =====
def save_frame_probs(stem, times, probs):
    out = Path(OUT_DIR) / f"{stem}_frame_probs.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time_sec","prob_chainsaw"])
        for t,p in zip(times, probs): w.writerow([f"{t:.3f}", f"{p:.6f}"])
    return str(out)

def save_events(stem, events):
    out = Path(OUT_DIR) / f"{stem}_events.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["start_sec","end_sec","peak_prob"])
        for s,e,pk in events: w.writerow([f"{s:.3f}", f"{e:.3f}", f"{pk:.6f}"])
    return str(out)

def save_summary_json(stem, payload):
    out = Path(OUT_DIR) / f"{stem}_summary.json"
    with open(out, "w") as f: json.dump(payload, f, indent=2)
    return str(out)

def save_confusion_csv(tp, tn, fp, fn, stem="dataset_confusion"):
    out = Path(OUT_DIR) / f"{stem}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "Pred_NEG", "Pred_POS"])
        w.writerow(["True_NEG", tn, fp])
        w.writerow(["True_POS", fn, tp])
    return str(out)

def save_perfile_csv(per_file, stem="dataset_per_file"):
    out = Path(OUT_DIR) / f"{stem}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path","label","clip_score","clip_pred","gt_events","ev_TP","ev_FP","ev_FN","ev_precision","ev_recall","ev_f1"])
        for row in per_file:
            ev = row.get("event_metrics")
            w.writerow([
                row["path"], row["label"], f'{row["clip_score"]:.6f}', row["clip_pred"],
                ";".join([f"{s}-{e}" for (s,e) in row["gt_events"]]) if row["gt_events"] else "",
                "" if ev is None else ev["TP"],
                "" if ev is None else ev["FP"],
                "" if ev is None else ev["FN"],
                "" if ev is None else f'{ev["precision"]:.6f}',
                "" if ev is None else f'{ev["recall"]:.6f}',
                "" if ev is None else f'{ev["f1"]:.6f}',
            ])
    return str(out)

def save_latex_metrics(clip_summ, ev_sum=None, stem="metrics_table"):
    out = Path(OUT_DIR) / f"{stem}.tex"
    with open(out, "w") as f:
        f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
        f.write("Metric & Acc & Prec & Rec & F1 & Spec\\\\\\midrule\n")
        f.write(f'Values & {clip_summ["accuracy"]:.3f} & {clip_summ["precision"]:.3f} & {clip_summ["recall"]:.3f} & {clip_summ["f1"]:.3f} & {clip_summ["specificity"]:.3f}\\\\\\bottomrule\n')
        f.write("\\end{tabular}\n")
        if clip_summ.get("roc_auc") is not None or clip_summ.get("pr_ap") is not None:
            f.write("% ROC-AUC: {:.3f}, PR-AUC(AP): {:.3f}\n".format(
                clip_summ.get("roc_auc") or 0.0, clip_summ.get("pr_ap") or 0.0))
        if ev_sum is not None:
            f.write("% Event IoU>=%.2f: P=%.3f R=%.3f F1=%.3f\n" % (IOU_THR, ev_sum["precision"], ev_sum["recall"], ev_sum["f1"]))
    return str(out)

# ===== Main =====
@torch.no_grad()
def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load model
    pth = Path(STATE_PTH).expanduser()
    assert pth.exists(), f"state_dict not found: {pth}"
    m = TinyCNN().eval()
    sd = torch.load(str(pth), map_location="cpu")
    m.load_state_dict(sd, strict=False)

    # Dummy forward
    y = m(torch.randn(1,1,64,100))
    print("OK: loaded state_dict. Dummy forward shape:", tuple(y.shape))

    feat = frontend(SR, N_MELS)

    # -------- Single-file mode --------
    if AUDIO_PATH:
        audio_path = Path(AUDIO_PATH)
        wav, sr0 = torchaudio.load(str(audio_path))
        if sr0 != SR:
            wav = torchaudio.functional.resample(wav, sr0, SR)
        wav = wav.mean(0, keepdim=True)

        t, p = slide_and_score(m, wav, SR, WIN_S, HOP_S, feat)
        print(f"Frames: {len(p)} | max={p.max():.3f} | mean={p.mean():.3f}")

        events = binarize_events(t, p, TH_ON, TH_OFF)

        print("Events:")
        if not events: print("  (none)")
        else:
            for s,e,pk in events:
                print(f"  {s:6.2f}s → {e:6.2f}s (peak={pk:.2f})")

        clip_score = float(p.max()) if len(p)>0 else 0.0
        decision = "CHAINSAW" if (clip_score>=THRESH) else "OTHER"
        print("Decision:", decision)

        stem = audio_path.stem
        fp_csv = save_frame_probs(stem, t, p)
        ev_csv = save_events(stem, events)
        summ = {
            "mode": "single_file",
            "config": {"SR":SR,"N_MELS":N_MELS,"WIN_S":WIN_S,"HOP_S":HOP_S,
                       "THRESH":THRESH,"TH_ON":TH_ON,"TH_OFF":TH_OFF},
            "file": str(audio_path),
            "clip_score": clip_score,
            "decision": decision,
            "n_events": len(events)
        }
        js_path = save_summary_json(stem, summ)
        print(f"Saved: {fp_csv}\nSaved: {ev_csv}\nSaved: {js_path}")

    # -------- Dataset evaluation (optional) --------
    if EVAL_CSV:
        csv_path = Path(EVAL_CSV)
        assert csv_path.exists(), f"Eval CSV not found: {csv_path}"

        rows = []
        with open(csv_path, "r", newline="") as f:
            for rec in csv.DictReader(f):
                path = (rec.get("path") or "").strip()
                if not path: continue
                label = int(rec.get("label","0"))
                events_str = (rec.get("events","") or "").strip()
                gt_events = []
                if events_str:
                    toks = [x.strip() for x in events_str.replace(",",";").split(";") if x.strip()]
                    for tok in toks:
                        try:
                            s,e = tok.split("-")
                            gt_events.append((float(s), float(e)))
                        except: pass
                rows.append((path, label, gt_events))

        y_true, y_score, y_pred = [], [], []
        ev_agg = {"TP":0,"FP":0,"FN":0}
        per_file = []

        for path, label, gt_ev in rows:
            wav, sr0 = torchaudio.load(path)
            if sr0 != SR:
                wav = torchaudio.functional.resample(wav, sr0, SR)
            wav = wav.mean(0, keepdim=True)
            t, p = slide_and_score(m, wav, SR, WIN_S, HOP_S, feat)

            clip_score = float(p.max()) if len(p)>0 else 0.0
            clip_pred  = int(clip_score >= THRESH)
            y_true.append(label); y_score.append(clip_score); y_pred.append(clip_pred)

            ev_metrics = None
            if len(gt_ev)>0:
                pr_events = binarize_events(t, p, TH_ON, TH_OFF)
                pr_simple = [(s,e) for (s,e,pk) in pr_events]
                evm = event_metrics(gt_ev, pr_simple, IOU_THR)
                ev_metrics = evm
                ev_agg["TP"] += evm["TP"]; ev_agg["FP"] += evm["FP"]; ev_agg["FN"] += evm["FN"]

            per_file.append({
                "path": path, "label": int(label),
                "clip_score": clip_score, "clip_pred": int(clip_pred),
                "gt_events": gt_ev, "event_metrics": ev_metrics
            })

        clip_summ = clip_metrics(y_true, y_score, THRESH)
        ev_sum = None
        if any(len(gt)>0 for _,_,gt in rows):
            prec = safe_div(ev_agg["TP"], ev_agg["TP"]+ev_agg["FP"])
            rec  = safe_div(ev_agg["TP"], ev_agg["TP"]+ev_agg["FN"])
            f1   = safe_div(2*prec*rec, (prec+rec)) if (prec+rec)>0 else 0.0
            ev_sum = {"TP":ev_agg["TP"], "FP":ev_agg["FP"], "FN":ev_agg["FN"],
                      "precision":prec, "recall":rec, "f1":f1, "iou_thr":IOU_THR}

        print("\n=== Clip-level (Chainsaw vs Other) ===")
        print(f"Threshold  = {clip_summ['threshold']:.2f}")
        print(f"Accuracy   = {clip_summ['accuracy']*100:5.2f}%")
        print(f"Precision  = {clip_summ['precision']*100:5.2f}%")
        print(f"Recall     = {clip_summ['recall']*100:5.2f}%")
        print(f"F1-score   = {clip_summ['f1']*100:5.2f}%")
        print(f"Specificity= {clip_summ['specificity']*100:5.2f}%")
        if clip_summ["roc_auc"] is not None:
            print(f"ROC-AUC    = {clip_summ['roc_auc']:.3f}")
        if clip_summ["pr_ap"] is not None:
            print(f"PR-AUC (AP)= {clip_summ['pr_ap']:.3f}")
        print(f"Confusion  = TP={clip_summ['TP']} TN={clip_summ['TN']} FP={clip_summ['FP']} FN={clip_summ['FN']}")

        # Save dataset artefacts in OUT_DIR
        per_csv = save_perfile_csv(per_file, "dataset_per_file")
        conf_csv = save_confusion_csv(clip_summ["TP"], clip_summ["TN"], clip_summ["FP"], clip_summ["FN"])
        tex = save_latex_metrics(clip_summ, ev_sum, "metrics_table")

        payload = {
            "mode": "dataset",
            "config": {"SR":SR,"N_MELS":N_MELS,"WIN_S":WIN_S,"HOP_S":HOP_S,
                       "THRESH":THRESH,"TH_ON":TH_ON,"TH_OFF":TH_OFF,"IOU_THR":IOU_THR},
            "clip_metrics": clip_summ,
            "event_metrics": ev_sum,
            "per_file": per_file
        }
        js = save_summary_json("dataset_eval", payload)
        print(f"\nSaved → {per_csv}\nSaved → {conf_csv}\nSaved → {tex}\nSaved → {js}")

if __name__ == "__main__":
    main()
