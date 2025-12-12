# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import traceback

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("Loan Approval Predictor — Decision Tree")
st.write("Enter applicant details and get a loan approval prediction. This app is robust to missing scaler/encoders and shows helpful diagnostics.")

# -------------------------
# Configuration - must match training
# -------------------------
# Numeric columns used when scaler was fit (Option A)
NUMERIC_COLS = [
    'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
    'cibil_score', 'residential_assets_value', 'commercial_assets_value',
    'luxury_assets_value', 'bank_asset_value'
]

# Default full feature order used during training (should match model.feature_names_in_ if available)
DEFAULT_FEATURE_ORDER = [
    'no_of_dependents','education','self_employed','income_annum','loan_amount',
    'loan_term','cibil_score','residential_assets_value','commercial_assets_value',
    'luxury_assets_value','bank_asset_value'
]

# -------------------------
# Load artifacts
# -------------------------
@st.cache_resource
def load_artifact(path):
    try:
        if os.path.exists(path):
            return joblib.load(path)
    except Exception:
        return None
    return None

model = load_artifact("decision_tree_model.joblib")
scaler = load_artifact("scaler.joblib")       # optional; if present, was fit only on NUMERIC_COLS
encoders = load_artifact("encoders.joblib")   # optional dict expected: {"education": le, "self_employed": le, "loan_status": le}

if model is None:
    st.error("Missing model file: 'decision_tree_model.joblib' not found in the app directory.\nPlease place the trained model file and reload.")
    st.stop()

# Attempt to get model feature names; else fallback to DEFAULT_FEATURE_ORDER
try:
    model_feature_names = list(model.feature_names_in_)
except Exception:
    model_feature_names = DEFAULT_FEATURE_ORDER.copy()

# Informational messages (non-blocking)
if scaler is None:
    st.info("No 'scaler.joblib' found. If your model requires scaled numeric inputs, predictions may differ. This app will continue using raw numeric values.")
else:
    st.success("Loaded 'scaler.joblib' (will apply only to numeric columns).")

if encoders is None:
    st.info("No 'encoders.joblib' found. Default categorical mapping will be used:\n - education: Graduate -> 1, Not Graduate -> 0\n - self_employed: Yes -> 1, No -> 0\n - loan_status encoder not available (app will display numeric class if missing).")
else:
    st.success(f"Loaded 'encoders.joblib' with keys: {list(encoders.keys())}")

# -------------------------
# Helpers
# -------------------------
def safe_encode_categorical(value, col_name):
    """
    Use saved encoder if available, otherwise fallback to sensible defaults.
    Returns numeric encoding (int).
    """
    # use encoders if provided
    if encoders and col_name in encoders:
        le = encoders[col_name]
        try:
            return int(le.transform([value])[0])
        except Exception:
            # try to map by class labels if transform fails
            mapping = {str(label): idx for idx, label in enumerate(le.classes_)}
            if str(value) in mapping:
                return int(mapping[str(value)])
            # fallback below
    # fallback mappings (case-insensitive)
    val = str(value).strip().lower()
    if col_name == "education":
        return 1 if val.startswith("grad") else 0
    if col_name == "self_employed":
        return 1 if val in ("yes", "y", "true", "1") else 0
    # for other categories, just try numeric cast
    try:
        return int(value)
    except Exception:
        return 0

def build_input_row(inputs: dict, feature_order=model_feature_names) -> pd.DataFrame:
    """
    Build a single-row DataFrame with columns in the 'feature_order' expected by the model.
    Categorical values will be encoded (using encoders if present).
    """
    row = {}
    # build values exactly for columns in feature_order
    for col in feature_order:
        if col == 'education':
            row[col] = safe_encode_categorical(inputs.get('education', 'Not Graduate'), 'education')
        elif col == 'self_employed':
            row[col] = safe_encode_categorical(inputs.get('self_employed', 'No'), 'self_employed')
        else:
            # numeric or unknown column -> use raw value if provided
            row[col] = inputs.get(col, 0)
    df_row = pd.DataFrame([row], columns=feature_order)
    return df_row

def validate_numeric_columns(df_row: pd.DataFrame):
    """
    Ensure numeric columns are numeric types and not NaN. Returns (ok:bool, message:str)
    """
    for c in NUMERIC_COLS:
        if c not in df_row.columns:
            return False, f"Missing numeric column: {c}"
        # coerce
        try:
            df_row[c] = pd.to_numeric(df_row[c], errors='coerce').astype(float)
        except Exception:
            return False, f"Column {c} cannot be converted to numeric"
        if pd.isna(df_row[c].iloc[0]):
            return False, f"Numeric column {c} is NaN after conversion. Please provide a numeric value."
    return True, "OK"

def interpret_prediction_numeric(pred_numeric):
    """
    Use encoders['loan_status'] if available; otherwise provide best-guess mapping.
    """
    # if encoders provided and have 'loan_status', use inverse_transform
    if encoders and 'loan_status' in encoders:
        try:
            le_target = encoders['loan_status']
            return le_target.inverse_transform([int(pred_numeric)])[0]
        except Exception:
            pass
    # fallback — try to match model.classes_ if they are strings
    try:
        if any(isinstance(c, str) for c in model.classes_):
            # then pred_numeric might already be string class, but typically it's numeric
            idx = list(model.classes_).index(pred_numeric) if pred_numeric in list(model.classes_) else None
            if idx is not None:
                return model.classes_[idx]
    except Exception:
        pass
    # Last resort mapping (common): 0 -> Approved, 1 -> Rejected may not hold; return numeric
    return str(pred_numeric)

# -------------------------
# UI: Input form
# -------------------------
with st.form("input_form"):
    st.subheader("Applicant details")
    no_of_dependents = st.number_input("Number of dependents", min_value=0, max_value=50, value=0, step=1)
    education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self employed?", options=["No", "Yes"])
    income_annum = st.number_input("Annual income (INR)", min_value=0, value=500000, step=10000, format="%d")
    loan_amount = st.number_input("Loan amount (INR)", min_value=0, value=500000, step=10000, format="%d")
    loan_term = st.number_input("Loan term (months)", min_value=1, value=12, step=1)
    cibil_score = st.number_input("CIBIL score", min_value=300, max_value=900, value=650, step=1)
    st.write("**Asset values (enter 0 if not applicable)**")
    residential_assets_value = st.number_input("Residential assets value (INR)", min_value=0, value=0, step=10000, format="%d")
    commercial_assets_value = st.number_input("Commercial assets value (INR)", min_value=0, value=0, step=10000, format="%d")
    luxury_assets_value = st.number_input("Luxury assets value (INR)", min_value=0, value=0, step=10000, format="%d")
    bank_asset_value = st.number_input("Bank asset value (INR)", min_value=0, value=0, step=10000, format="%d")

    submitted = st.form_submit_button("Predict")

# -------------------------
# Prediction & Display
# -------------------------
if submitted:
    # collect inputs into dict keyed by expected feature names where possible
    inputs = {
        'no_of_dependents': no_of_dependents,
        'education': education,
        'self_employed': self_employed,
        'income_annum': income_annum,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value,
        'commercial_assets_value': commercial_assets_value,
        'luxury_assets_value': luxury_assets_value,
        'bank_asset_value': bank_asset_value
    }

    # Build row DataFrame in the model's feature order
    try:
        row_df = build_input_row(inputs, feature_order=model_feature_names)
    except Exception as e:
        st.error(f"Failed to build input row: {e}")
        st.stop()

    # Validate numeric columns
    ok, msg = validate_numeric_columns(row_df)
    if not ok:
        st.error(f"Invalid numeric inputs: {msg}")
        st.stop()

    # Keep a copy of the pre-scaled row for debugging
    row_before_scaling = row_df.copy(deep=True)

    # Apply scaler only to NUMERIC_COLS if available
    row_after_scaling = None
    if scaler is not None:
        try:
            row_df[NUMERIC_COLS] = scaler.transform(row_df[NUMERIC_COLS])
            row_after_scaling = row_df.copy(deep=True)
        except Exception as e:
            st.error(f"Error applying scaler: {e}\nMake sure 'scaler.joblib' was fitted on these numeric columns (in same order): {NUMERIC_COLS}")
            st.stop()

    # Predict using DataFrame (keeps feature names)
    try:
        pred_numeric = model.predict(row_df)[0]  # numeric label as used by model
    except Exception as e:
        st.error(f"Prediction failed: {e}\nTrace:\n{traceback.format_exc()}")
        st.stop()

    # Get probability vector if available
    proba = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(row_df)[0]
    except Exception:
        proba = None

    # Interpret predicted label (use saved target encoder if present)
    human_label = interpret_prediction_numeric(pred_numeric)

    # Display result
    st.markdown("### Prediction")
    st.success(f"**Result:** {human_label}")

    # Show probability details: map model.classes_ to readable labels using target encoder if available
    if proba is not None:
        # make readable class labels
        readable_classes = []
        if encoders and 'loan_status' in encoders:
            try:
                le_target = encoders['loan_status']
                for cls in model.classes_:
                    # cls are numeric ids (0/1) — inverse transform to human label
                    readable_classes.append(le_target.inverse_transform([int(cls)])[0])
            except Exception:
                readable_classes = [str(c) for c in model.classes_]
        else:
            # model.classes_ might be numeric or strings — show raw
            readable_classes = [str(c) for c in model.classes_]

        # Find index for numeric prediction in model.classes_
        try:
            pred_idx = list(model.classes_).index(pred_numeric)
            st.write(f"**Confidence ({human_label}):** {proba[pred_idx]:.4f}")
        except Exception:
            st.write("**Class probabilities (full breakdown):**")
            for cls, rcls, p in zip(model.classes_, readable_classes, proba):
                st.write(f"  {rcls} ({cls}) : {p:.4f}")
    else:
        st.info("Model does not provide probability estimates (predict_proba not available).")

    # Display input summary (readable) in a two-column table to avoid mixed-type Arrow errors
    display_features = [
        'no_of_dependents','education','self_employed','income_annum','loan_amount',
        'loan_term','cibil_score','residential_assets_value','commercial_assets_value',
        'luxury_assets_value','bank_asset_value'
    ]
    display_values = [
        str(no_of_dependents),
        education,
        self_employed,
        f"{income_annum:,}",
        f"{loan_amount:,}",
        str(loan_term),
        str(cibil_score),
        f"{residential_assets_value:,}",
        f"{commercial_assets_value:,}",
        f"{luxury_assets_value:,}",
        f"{bank_asset_value:,}"
    ]
    df_display = pd.DataFrame({"feature": display_features, "value": display_values})
    st.markdown("### Input summary")
    st.table(df_display)

    # Debug panel - expandable
    with st.expander("Show internals / debug info (toggle)"):
        st.write("**Model feature names (what the model expects):**")
        st.write(model_feature_names)

        st.write("**Model.classes_ (raw):**")
        st.write(list(model.classes_))

        if encoders and 'loan_status' in encoders:
            try:
                st.write("**encoders['loan_status'].classes_ (readable):**")
                st.write(list(encoders['loan_status'].classes_))
            except Exception:
                st.write("encoders['loan_status'] present but couldn't display classes.")
        else:
            st.write("No saved loan_status encoder found in encoders.joblib")

        st.write("**Row BEFORE scaling (what app encoded):**")
        st.dataframe(row_before_scaling.T)

        if row_after_scaling is not None:
            st.write("**Row AFTER scaling (numeric columns transformed):**")
            st.dataframe(row_after_scaling.T)
        else:
            st.write("No scaler used - showing pre-scaled row again:")
            st.dataframe(row_before_scaling.T)

        st.write("**Prediction (raw numeric) and full proba vector if present:**")
        st.write({"pred_numeric": int(pred_numeric), "proba_vector": proba.tolist() if proba is not None else None})

        st.write("**Readable mapping for model.classes_ (if loan_status encoder present):**")
        try:
            if encoders and 'loan_status' in encoders:
                le_target = encoders['loan_status']
                mapping = {str(cls): le_target.inverse_transform([int(cls)])[0] for cls in model.classes_}
                st.write(mapping)
            else:
                st.write("No target encoder to map numeric classes to readable labels.")
        except Exception as e:
            st.write("Failed to build readable mapping:", str(e))

# -------------------------
# Footer / tips
# -------------------------
st.markdown("---")
st.caption(
    "Tips:\n"
    "- This app applies the saved StandardScaler only to the numeric columns listed in NUMERIC_COLS.\n"
    "- Ensure 'scaler.joblib' was fitted on these exact numeric columns (same names & order) if you want scaling.\n"
    "- Keep 'encoders.joblib' containing 'education', 'self_employed', and 'loan_status' to ensure consistent mapping.\n"
    "- For production reliability, save & load a full sklearn Pipeline (preprocessor + model) and call pipeline.predict(df_row)."
)
