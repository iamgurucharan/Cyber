import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
inp = ROOT / 'tools' / 'batch_results.json'
out_dir = ROOT / 'reports'
out_dir.mkdir(parents=True, exist_ok=True)

with open(inp, encoding='utf-8') as f:
    data = json.load(f)

probs = []
labels = []
for r in data:
    p = r.get('risk_score_not_secure_proba')
    if p is None:
        continue
    probs.append(float(p))
    labels.append(r.get('predicted_label'))

probs = np.array(probs)

summary = {
    'n': int(len(probs)),
    'mean': float(np.mean(probs)) if probs.size else None,
    'median': float(np.median(probs)) if probs.size else None,
    'std': float(np.std(probs, ddof=1)) if probs.size>1 else None,
    'min': float(np.min(probs)) if probs.size else None,
    'max': float(np.max(probs)) if probs.size else None,
    'percentiles': {
        '10': float(np.percentile(probs,10)) if probs.size else None,
        '25': float(np.percentile(probs,25)) if probs.size else None,
        '50': float(np.percentile(probs,50)) if probs.size else None,
        '75': float(np.percentile(probs,75)) if probs.size else None,
        '90': float(np.percentile(probs,90)) if probs.size else None,
    }
}

# thresholds to evaluate
thresholds = [0.5, 0.6, 0.7, 0.8]
threshold_stats = {}
for t in thresholds:
    flagged = int(np.sum(probs >= t))
    pct = float(100.0 * flagged / len(probs)) if len(probs)>0 else 0.0
    threshold_stats[str(t)] = {'flagged_count': flagged, 'flagged_percent': pct}

# suggested threshold: choose where tail > 75th percentile? Suggest t = median + std
suggested = None
if probs.size>0:
    # prefer a threshold at 75th percentile or median+std, whichever is higher but <=0.9
    p75 = float(np.percentile(probs,75))
    med = float(np.median(probs))
    std = float(np.std(probs, ddof=1)) if probs.size>1 else 0.0
    alt = min(max(med + std, med), 0.9)
    suggested = round(max(p75, alt), 3)

# Histogram
plt.figure(figsize=(6,4))
plt.hist(probs, bins=10, color='tab:blue', edgecolor='white')
plt.xlabel('P(Not Secure)')
plt.ylabel('Count')
plt.title('Batch predictions distribution')
for t in thresholds:
    plt.axvline(t, color='red', linestyle='--', linewidth=0.8)
plt.tight_layout()
hist_path = out_dir / 'batch_predictions_hist.png'
plt.savefig(hist_path, dpi=150)
plt.close()

report = {
    'summary': summary,
    'threshold_stats': threshold_stats,
    'suggested_threshold': suggested,
    'n_flagged_at_suggested': int(np.sum(probs >= (suggested if suggested is not None else 0.0))) if probs.size else 0,
    'histogram': str(hist_path.relative_to(ROOT))
}

out_path = out_dir / 'batch_report.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2)

# also write a short human-readable TXT
txt = out_dir / 'batch_report.txt'
with open(txt, 'w', encoding='utf-8') as f:
    f.write('Batch predictions analysis\n')
    f.write('==========================\n\n')
    f.write(f"Count: {summary['n']}\n")
    f.write(f"Mean P(Not Secure): {summary['mean']:.3f}\n")
    f.write(f"Median: {summary['median']:.3f}\n")
    f.write(f"Std: {summary['std']:.3f}\n")
    f.write(f"Min: {summary['min']:.3f}, Max: {summary['max']:.3f}\n\n")
    f.write('Percentiles:\n')
    for k,v in summary['percentiles'].items():
        f.write(f"  {k}%: {v:.3f}\n")
    f.write('\nThreshold evaluation:\n')
    for k,v in threshold_stats.items():
        f.write(f"  >={k}: {v['flagged_count']} sites ({v['flagged_percent']:.1f}%)\n")
    f.write('\n')
    f.write(f"Suggested threshold (heuristic): {suggested}\n")
    f.write(f"Sites flagged at suggested threshold: {report['n_flagged_at_suggested']}\n")
    f.write(f"Histogram image: {report['histogram']}\n")

print('Wrote report to', out_path, 'and', txt, 'histogram:', hist_path)
print(json.dumps(report, indent=2))
