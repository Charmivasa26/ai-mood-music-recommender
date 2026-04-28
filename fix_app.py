"""
fix_app.py
Run this ONCE in your mood_music_app folder:
    python fix_app.py

It will patch app.py directly on your machine.
"""

import re

print("Reading app.py...")
with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

original = content
fixes = 0

# ── FIX 1: engineer_features function (add if missing) ───────
if "def engineer_features" not in content:
    old = "def predict_mood("
    new = '''def engineer_features(fv):
    """Compute the 8 extra features train_model_v3.py added."""
    fv = dict(fv)
    fv['energy_sq']           = fv['energy'] ** 2
    fv['dance_energy']        = fv['danceability'] * fv['energy']
    fv['acoustic_inv_energy'] = fv['acousticness'] * (1 - fv['energy'])
    fv['valence_dance']       = fv['valence'] * fv['danceability']
    fv['high_energy_flag']    = 1.0 if fv['energy'] > 0.8 else 0.0
    fv['acoustic_flag']       = 1.0 if fv['acousticness'] > 0.7 else 0.0
    fv['groovy_flag']         = 1.0 if fv['danceability'] > 0.7 else 0.0
    fv['instr_flag']          = 1.0 if fv['instrumentalness'] > 0.5 else 0.0
    return fv


def predict_mood('''
    content = content.replace(old, new, 1)
    fixes += 1
    print("  ✅ Fix 1: engineer_features function added")
else:
    print("  ✓  Fix 1: engineer_features already present")

# ── FIX 2: predict_mood body — always engineer, use ACTIVE cols ──
old2 = re.compile(
    r'def predict_mood\(.*?\n(.*?)'      # function def
    r'(    arr = np\.array\(\[\[.*?for c in .*?FEATURE_COLS.*?\]\]\))',
    re.DOTALL
)

# Simpler targeted replace: fix the arr = line inside predict_mood
# Find the function and replace its body
pm_start = content.find("def predict_mood(")
pm_end   = content.find("\ndef ", pm_start + 1)
pm_block = content[pm_start:pm_end]

new_pm_block = '''def predict_mood(feature_values, model, scaler, le, feat_cols=None):
    """Always engineers all 17 features before calling scaler."""
    if feat_cols is None:
        feat_cols = ACTIVE_FEATURE_COLS
    feature_values = engineer_features(feature_values)
    arr = np.array([[feature_values.get(c, 0.0) for c in feat_cols]])
    arr_scaled = scaler.transform(arr)
    arr_reshaped = arr_scaled.reshape(1, len(feat_cols), 1)
    probs = model.predict(arr_reshaped, verbose=0)[0]
    pred_idx = np.argmax(probs)
    pred_label = le.inverse_transform([pred_idx])[0]
    prob_dict = {le.inverse_transform([i])[0]: float(probs[i]) for i in range(len(probs))}
    return pred_label, prob_dict
'''

content = content[:pm_start] + new_pm_block + content[pm_end:]
fixes += 1
print("  ✅ Fix 2: predict_mood body rewritten")

# ── FIX 3: All predict_mood() call sites — pass ACTIVE_FEATURE_COLS ──
n = 0
for old_call, new_call in [
    ("predict_mood(feature_values, model, scaler, le)",
     "predict_mood(feature_values, model, scaler, le, ACTIVE_FEATURE_COLS)"),
    ("predict_mood(song_feats, model, scaler, le)",
     "predict_mood(song_feats, model, scaler, le, ACTIVE_FEATURE_COLS)"),
    # handle case where it already has feat_cols=None or FEATURE_COLS
    ("predict_mood(feature_values, model, scaler, le, FEATURE_COLS)",
     "predict_mood(feature_values, model, scaler, le, ACTIVE_FEATURE_COLS)"),
    ("predict_mood(song_feats, model, scaler, le, FEATURE_COLS)",
     "predict_mood(song_feats, model, scaler, le, ACTIVE_FEATURE_COLS)"),
]:
    if old_call in content:
        content = content.replace(old_call, new_call)
        n += 1
fixes += 1
print(f"  ✅ Fix 3: {n} predict_mood call site(s) updated")

# ── FIX 4: Radar chart — the line 566 bug ────────────────────
radar_patterns = [
    # pattern 1 — original
    "norm_vals = scaler.transform([[song_feats[c] for c in FEATURE_COLS]])[0]",
    # pattern 2 — partially fixed
    "norm_vals = scaler.transform([[song_feats[c] for c in ACTIVE_FEATURE_COLS]])[0]",
]
radar_fixed = False
for pat in radar_patterns:
    if pat in content:
        content = content.replace(
            pat,
            "song_feats_eng = engineer_features(song_feats)\n"
            "            norm_vals = scaler.transform([[song_feats_eng.get(c, 0.0) for c in ACTIVE_FEATURE_COLS]])[0]"
        )
        radar_fixed = True
        break

# Also fix the feat_idx line that follows
content = content.replace(
    "feat_idx = [FEATURE_COLS.index(f) for f in feats_display]",
    "feat_idx = [ACTIVE_FEATURE_COLS.index(f) for f in feats_display if f in ACTIVE_FEATURE_COLS]"
)

if radar_fixed:
    fixes += 1
    print("  ✅ Fix 4: Radar chart scaler call fixed")
else:
    print("  ✓  Fix 4: Radar chart already fixed or pattern not found")

# ── FIX 5: ACTIVE_FEATURE_COLS fallback at top level ─────────
if "ACTIVE_FEATURE_COLS = FEATURE_COLS" not in content:
    # Add a safe fallback right after FEATURE_COLS definition
    content = content.replace(
        "FEATURE_RANGES = {",
        "ACTIVE_FEATURE_COLS = FEATURE_COLS  # will be overwritten by load_artifacts\n\nFEATURE_RANGES = {"
    )
    fixes += 1
    print("  ✅ Fix 5: ACTIVE_FEATURE_COLS fallback added")
else:
    print("  ✓  Fix 5: ACTIVE_FEATURE_COLS fallback already present")

# ── SAVE ─────────────────────────────────────────────────────
if content != original:
    # backup first
    with open("app_backup.py", "w", encoding="utf-8") as f:
        f.write(original)
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n✅ app.py patched successfully ({fixes} fixes applied)")
    print("   Original saved as app_backup.py")
else:
    print("\n⚠️  No changes made — file may already be correct")

print("\nNow run:  streamlit run app.py")