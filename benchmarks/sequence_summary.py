"""Describe paired per-sequence scores from the manuscript's hashed receipts."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
generator = HERE / 'generate_table.py'
with contextlib.redirect_stdout(io.StringIO()):
    source = runpy.run_path(str(generator))
rows = source['rows']
if [name for name, _ in rows] != ['Cracker', 'Mustard', 'Bleach', 'Sugar', 'Tomato', 'Bleach-hard*']:
    raise ValueError('unexpected manuscript sequence selection')
items = []
for name, result in rows:
    p, b = result['point2pose'], result['bundletrack']
    n = p['denominator_excluding_initial']
    if type(n) is not int or n <= 0 or n != b['denominator_excluding_initial']:
        raise ValueError('invalid paired denominator')
    for method in (p, b):
        if not 0 <= method['adds_le_10mm_count'] <= n or not 0 <= method['recall_auc_0_to_100mm'] <= 1:
            raise ValueError('invalid score')
    items.append({'sequence': name, 'n': n,
                  'point2pose_success_fraction': p['adds_le_10mm_count'] / n,
                  'bundletrack_success_fraction': b['adds_le_10mm_count'] / n,
                  'point2pose_auc': p['recall_auc_0_to_100mm'],
                  'bundletrack_auc': b['recall_auc_0_to_100mm']})
selected = items[:5]
avg = lambda key: sum(row[key] for row in selected) / len(selected)
weighted = lambda key: sum(row[key] * row['n'] for row in selected) / sum(row['n'] for row in selected)
out = {'scope': 'post-hoc descriptive aggregate of five selected development sequences',
       'source_generator_sha256': hashlib.sha256(generator.read_bytes()).hexdigest(),
       'sequence_count': len(selected), 'post_initial_frames': sum(r['n'] for r in selected),
       'equal_sequence_weighted': {key: avg(key) for key in ('point2pose_success_fraction', 'bundletrack_success_fraction', 'point2pose_auc', 'bundletrack_auc')},
       'frame_weighted': {key: weighted(key) for key in ('point2pose_success_fraction', 'bundletrack_success_fraction')},
       'selected_sequences': selected, 'subsequently_selected_not_in_aggregate': items[5],
       'limits': ['selected correlated development videos are not random independent trials',
                  '10 mm threshold is post-hoc and uses reference masks and first-frame gauge alignment',
                  'no confidence intervals, general population claims, or physical truth are established']}
print(json.dumps(out, indent=2, allow_nan=False))
