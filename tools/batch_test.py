import sys
from pathlib import Path
import time
import json

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.predictor import RiskPredictor
from backend.website_scanner import scan_url
from backend.burp_integration import merge_burp_into_features

urls = [
    "https://example.com",
    "https://www.google.com",
    "https://www.youtube.com",
    "https://www.facebook.com",
    "https://www.amazon.com",
    "https://www.yahoo.com",
    "https://www.reddit.com",
    "https://www.instagram.com",
    "https://www.linkedin.com",
    "https://www.netflix.com",
    "https://www.bing.com",
    "https://www.pinterest.com",
    "https://www.ebay.com",
    "https://www.twitch.tv",
    "https://www.walmart.com",
    "https://wordpress.com",
    "https://www.apple.com",
    "https://www.microsoft.com",
    "https://github.com",
    "https://stackoverflow.com",
    "https://medium.com",
    "https://www.cnn.com",
    "https://www.bbc.co.uk",
    "https://www.nytimes.com",
    "https://www.theguardian.com",
    "https://www.paypal.com",
    "https://www.adobe.com",
    "https://www.dropbox.com",
    "https://slack.com",
    "https://discord.com",
    "https://www.spotify.com",
    "https://www.quora.com",
    "https://news.ycombinator.com",
    "https://www.etsy.com",
    "https://www.salesforce.com",
    "https://www.shopify.com",
    "https://www.mozilla.org",
    "https://www.oracle.com",
    "https://www.wix.com",
    "https://www.ikea.com",
    "https://www.imdb.com",
    "https://www.tripadvisor.com",
    "https://www.target.com",
    "https://www.zillow.com",
    "https://www.craigslist.org",
    "https://www.behance.net",
    "https://www.bloomberg.com",
    "https://www.healthline.com",
    "https://www.nih.gov",
]

p = RiskPredictor()
try:
    p.load()
except Exception as e:
    print('Model load error:', e)
    raise

results = []
for u in urls:
    try:
        scan = scan_url(u)
        merged = merge_burp_into_features(scan, simulate_seed=123)
        feats = merged["features"]
        risk, pred_class, proba = p.predict_proba_row(feats)
        label = "Not Secure" if pred_class == 1 else "Secure"
        results.append({
            "url": u,
            "final_url": scan.final_url,
            "scan_error": scan.error,
            "predicted_label": label,
            "risk_score_not_secure_proba": float(risk),
            "class_probs": [float(p) for p in proba],
        })
    except Exception as e:
        results.append({"url": u, "error": str(e)})
    time.sleep(1)

out = Path(__file__).resolve().parent / "batch_results.json"
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print('Wrote', out)
print(json.dumps(results, indent=2))
