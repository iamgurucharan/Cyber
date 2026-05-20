from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from backend.predictor import RiskPredictor
from backend.website_scanner import scan_url
from backend.burp_integration import merge_burp_into_features
from backend.app import RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MED
p=RiskPredictor(); p.load()
scan=scan_url('https://example.com')
merged=merge_burp_into_features(scan, simulate_seed=123)
feats=merged['features']
risk, pred_class, proba = p.predict_proba_row(feats)
print('risk', risk)
print('thresholds', RISK_THRESHOLD_MED, RISK_THRESHOLD_HIGH)
if risk>=RISK_THRESHOLD_HIGH:
    print('High')
elif risk>=RISK_THRESHOLD_MED:
    print('Medium')
else:
    print('Low')
