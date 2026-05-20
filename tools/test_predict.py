import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.predictor import RiskPredictor
from backend.website_scanner import scan_url
from backend.burp_integration import merge_burp_into_features

p = RiskPredictor()
try:
    p.load()
except Exception as e:
    print('load_error', e)
    raise

scan = scan_url('https://example.com')
b = merge_burp_into_features(scan, simulate_seed=12345)
feats = b['features']
print('features_sample:', {k: feats.get(k) for k in ['ssl_valid','has_hsts','burp_scan_score','burp_issues_high']})

risk, pred_class, proba = p.predict_proba_row(feats)
print('predicted_class:', pred_class)
print('risk_score_not_secure_proba:', risk)
print('class_probabilities:', proba)
print('predicted_label:', 'Not Secure' if pred_class==1 else 'Secure')
