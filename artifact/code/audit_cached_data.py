#!/usr/bin/env python3
"""Audit the legacy cache without treating it as a forecasting dataset."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'code/polymarket_cache/polymarket_data.json'


def audit(path=SOURCE):
    raw = path.read_bytes()
    markets = json.loads(raw)
    checks = []
    for m in markets:
        increments, volumes = m.get('delta_x', []), m.get('volumes', [])
        proxy = len(increments) == len(volumes) and all(
            abs(float(v) - (100 * abs(float(z)) + 1)) < 1e-8
            for z, v in zip(increments, volumes)
        )
        checks.append({
            'market_id': m.get('id'), 'question': m.get('question'),
            'cached_outcome': m.get('outcome'), 'n_prices': len(m.get('prices', [])),
            'volume_equals_price_change_proxy': proxy,
            'has_timestamps': 'timestamps' in m,
            'has_event_group': 'event_id' in m,
            'has_resolution_timestamp': 'resolution_timestamp' in m,
        })
    return {
        'source': str(path.relative_to(ROOT)),
        'sha256': hashlib.sha256(raw).hexdigest(),
        'n_markets': len(markets),
        'outcome_counts': dict(Counter(str(m.get('outcome')) for m in markets)),
        'all_volumes_are_price_change_proxy': all(c['volume_equals_price_change_proxy'] for c in checks),
        'markets_with_timestamps': sum(c['has_timestamps'] for c in checks),
        'suitable_for_price_volume_forecast_validation': False,
        'reasons': [
            'Interval volume is reconstructed from absolute logit price changes, not measured trades.',
            'The cache omits observation timestamps, preventing pre-resolution horizon checks.',
            'The cache omits event groups and resolution timestamps.',
            'The legacy script filters out boundary prices and selects by resolved status and total volume.',
            'Cached outcomes are inferred from terminal token prices; no independent settlement record is cached.',
        ],
        'markets': checks,
    }


if __name__ == '__main__':
    result = audit()
    target = ROOT / 'revision/cache_audit.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'markets'}, indent=2))
