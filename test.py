# test_harness.py
import joblib
import pandas as pd
import numpy as np

# --- config: match your app/training ---
NUMERIC_COLS = [
    'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
    'cibil_score', 'residential_assets_value', 'commercial_assets_value',
    'luxury_assets_value', 'bank_asset_value'
]

FEATURE_ORDER = [
    'no_of_dependents','education','self_employed','income_annum','loan_amount',
    'loan_term','cibil_score','residential_assets_value','commercial_assets_value',
    'luxury_assets_value','bank_asset_value'
]

# --- load artifacts ---
model = joblib.load("decision_tree_model.joblib")
print("Loaded model. model.classes_:", model.classes_)
scaler = None
encoders = None
try:
    scaler = joblib.load("scaler.joblib")
    print("Loaded scaler (will apply to numeric cols).")
except Exception:
    print("No scaler found (skipping scaling).")
try:
    encoders = joblib.load("encoders.joblib")
    print("Loaded encoders keys:", list(encoders.keys()))
except Exception:
    print("No encoders file found.")

# helper to encode categories same as app
def encode_cat(value, col_name):
    if encoders and col_name in encoders:
        try:
            return int(encoders[col_name].transform([value])[0])
        except Exception:
            pass
    if col_name == "education":
        return 1 if str(value).lower().startswith("grad") else 0
    if col_name == "self_employed":
        return 1 if str(value).lower() in ("yes","y","true","1") else 0
    return value

def make_row(no_of_dependents=0, education="Graduate", self_employed="No",
             income_annum=500000, loan_amount=5000000, loan_term=12,
             cibil_score=650, residential_assets_value=0,
             commercial_assets_value=0, luxury_assets_value=0, bank_asset_value=0):
    row = {
        'no_of_dependents': no_of_dependents,
        'education': encode_cat(education, 'education'),
        'self_employed': encode_cat(self_employed, 'self_employed'),
        'income_annum': income_annum,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value,
        'commercial_assets_value': commercial_assets_value,
        'luxury_assets_value': luxury_assets_value,
        'bank_asset_value': bank_asset_value
    }
    df = pd.DataFrame([row], columns=FEATURE_ORDER)
    # ensure numeric types
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
    return df

def interpret_pred(pred_numeric):
    if encoders and 'loan_status' in encoders:
        try:
            return encoders['loan_status'].inverse_transform([int(pred_numeric)])[0]
        except Exception:
            pass
    # fallback: use model.classes_ if strings or show numeric
    if any(isinstance(c, str) for c in model.classes_):
        return str(pred_numeric)
    return str(pred_numeric)

# --- Single test example (your approved-training row) ---
example = make_row(
    no_of_dependents=2,
    education="Graduate",
    self_employed="No",
    income_annum=9600000,
    loan_amount=29900000,
    loan_term=12,
    cibil_score=778,
    residential_assets_value=2400000,
    commercial_assets_value=17600000,
    luxury_assets_value=22700000,
    bank_asset_value=8000000
)

# apply scaler only to numeric cols if available
if scaler is not None:
    example[NUMERIC_COLS] = scaler.transform(example[NUMERIC_COLS])

pred = model.predict(example)[0]
proba = model.predict_proba(example)[0] if hasattr(model, "predict_proba") else None
print("=== Single example ===")
print("Row (post-scale if applied):")
print(example.T)
print("Pred raw:", pred, "->", interpret_pred(pred))
if proba is not None:
    print("Prob vector (classes order):", list(zip(model.classes_, proba)))
print()

# --- Sweep CIBIL values while keeping income and loan_amount as you asked ---
print("=== Sweep CIBIL (keeping income and loan_amount fixed) ===")
fixed_income = 50            # set this to the 'income' you want to test (units same as training data)
fixed_loan_amount = 5000000  # the loan amount you want to test
for cibil in [300, 400, 500, 600, 650, 700, 750, 778, 800, 850, 900]:
    row = make_row(
        no_of_dependents=0,
        education="Graduate",
        self_employed="No",
        income_annum=fixed_income,
        loan_amount=fixed_loan_amount,
        loan_term=12,
        cibil_score=cibil,
        residential_assets_value=0,
        commercial_assets_value=0,
        luxury_assets_value=0,
        bank_asset_value=0
    )
    if scaler is not None:
        row[NUMERIC_COLS] = scaler.transform(row[NUMERIC_COLS])
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0] if hasattr(model, "predict_proba") else None
    readable = interpret_pred(pred)
    if proba is not None:
        # find index of pred in model.classes_ and show its prob
        idx = list(model.classes_).index(pred)
        prob_for_pred = proba[idx]
        print(f"CIBIL={cibil:3d} -> {readable:9s} (prob={prob_for_pred:.4f}) raw_pred={pred}")
    else:
        print(f"CIBIL={cibil:3d} -> {readable} (no probs)")

# --- Optional: Sweep income while holding CIBIL high to test if income matters ---
print("\n=== Sweep Income (keep CIBIL=778) ===")
for inc in [0, 50, 100000, 1000000, 5000000, 9600000]:
    row = make_row(
        no_of_dependents=0,
        education="Graduate",
        self_employed="No",
        income_annum=inc,
        loan_amount=5000000,
        loan_term=12,
        cibil_score=778,
        residential_assets_value=0,
        commercial_assets_value=0,
        luxury_assets_value=0,
        bank_asset_value=0
    )
    if scaler is not None:
        row[NUMERIC_COLS] = scaler.transform(row[NUMERIC_COLS])
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0] if hasattr(model, "predict_proba") else None
    readable = interpret_pred(pred)
    if proba is not None:
        idx = list(model.classes_).index(pred)
        print(f"Income={inc:10d} -> {readable:9s} (prob={proba[idx]:.4f})")
    else:
        print(f"Income={inc:10d} -> {readable}")
