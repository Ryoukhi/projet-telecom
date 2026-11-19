"""Inspecte les features attendues par le modèle."""

import joblib

model = joblib.load("models/model.pkl")
print("Type:", type(model).__name__)

# XGBoost stocke les noms de features dans feature_names_in_ (sklearn API)
if hasattr(model, "feature_names_in_"):
    print("Features attendues (feature_names_in_):")
    for i, f in enumerate(model.feature_names_in_):
        print(f"  {i}: {f}")
elif hasattr(model, "get_booster"):
    booster = model.get_booster()
    print("Features attendues (booster.feature_names):")
    for i, f in enumerate(booster.feature_names):
        print(f"  {i}: {f}")
else:
    print("Pas de noms de features stockés. Le modèle attend n_features_in_:")
    print("  n_features:", model.n_features_in_)
