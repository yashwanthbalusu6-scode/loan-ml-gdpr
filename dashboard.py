"""Streamlit frontend for the loan-approval system."""
import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st


_DEFAULT_BACKEND = "https://yashjanthb-loan-ml-api.hf.space"
BACKEND_URL = os.getenv("BACKEND_URL", _DEFAULT_BACKEND).rstrip("/")

st.set_page_config(page_title="Loan ML + GDPR", page_icon="🏦", layout="wide")

st.markdown(
    """
    <style>
    .block-container { max-width: 1200px; padding-top: 2rem; }
    .decision-card {
        padding: 1.25rem 1.5rem; border-radius: 12px; font-size: 1.4rem;
        font-weight: 700; text-align: center; margin: 1rem 0;
    }
    .approved { background: #d1fae5; color: #065f46; border-left: 6px solid #10b981; }
    .denied { background: #fee2e2; color: #991b1b; border-left: 6px solid #ef4444; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str, **params):
    r = requests.get(f"{BACKEND_URL}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def api_post(path: str, json: dict):
    r = requests.post(f"{BACKEND_URL}{path}", json=json, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code}: {r.text}")
    return r.json()


st.title("🏦 Loan ML + GDPR Compliance")
st.caption(f"Backend: `{BACKEND_URL}`")

with st.sidebar:
    st.header("Status")
    try:
        h = api_get("/health")
        st.success(f"API: {h.get('status')} · model loaded: {h.get('model_loaded')}")
    except Exception as e:
        st.error(f"API unreachable: {e}")


tab_predict, tab_explain, tab_gdpr, tab_metrics = st.tabs(
    ["🔮 Predict", "📊 Explain", "🛡️ GDPR Compliance", "📈 Model Metrics"]
)


with tab_predict:
    st.subheader("Apply for a loan decision")
    c = st.columns(5)
    age = c[0].number_input("Age", 18, 100, 35)
    income = c[1].number_input("Income (USD)", 0, 1_000_000, 75_000, step=1_000)
    credit_score = c[2].number_input("Credit score", 300, 850, 720)
    loan_amount = c[3].number_input("Loan amount (USD)", 0, 1_000_000, 25_000, step=1_000)
    employment_years = c[4].number_input("Employment (yrs)", 0, 70, 5)
    customer_id_input = st.text_input("Customer ID (optional)", placeholder="auto-generated if empty")

    if st.button("Predict", type="primary"):
        payload = {
            "age": int(age),
            "income": float(income),
            "credit_score": int(credit_score),
            "loan_amount": float(loan_amount),
            "employment_years": int(employment_years),
        }
        if customer_id_input.strip():
            payload["customer_id"] = customer_id_input.strip()
        try:
            res = api_post("/predict", payload)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
        else:
            cls = "approved" if res["prediction"] == 1 else "denied"
            st.markdown(
                f'<div class="decision-card {cls}">{res["decision_text"]} '
                f'· confidence {res["probability"] * 100:.1f}%</div>',
                unsafe_allow_html=True,
            )
            st.write(f"**Customer ID:** `{res['customer_id']}`")
            st.write(f"**Why:** {res['explanation_text']}")

            shap_df = pd.DataFrame(res["shap_values"]).set_index("feature")[["shap"]]
            st.bar_chart(shap_df)
            st.caption("SHAP: positive pushes toward APPROVED, negative toward DENIED.")
            with st.expander("Raw response"):
                st.json(res)


with tab_explain:
    st.subheader("Re-examine a past decision")
    cid = st.text_input("Customer ID", key="explain_cid")
    if st.button("Look up explanation"):
        try:
            data = api_post("/explain", {"customer_id": cid})
        except Exception as e:
            st.error(str(e))
        else:
            st.write(
                f"**Decision:** {'APPROVED' if data['prediction'] == 1 else 'DENIED'} "
                f"· probability {data['probability'] * 100:.1f}%"
            )
            shap = data["shap"]
            df = pd.DataFrame({
                "feature": shap["feature_names"],
                "shap": shap["shap_values"],
            }).set_index("feature")
            st.bar_chart(df)
            with st.expander("Full record"):
                st.json(data)


with tab_gdpr:
    st.subheader("Audit log")
    cf = st.columns([2, 1, 1])
    filt_cid = cf[0].text_input("Filter by customer_id", key="audit_cid")
    if cf[1].button("Refresh"):
        st.rerun()

    try:
        rows = api_get("/audit-log", customer_id=filt_cid or None, limit=100)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No audit entries match your filter.")
    except Exception as e:
        st.error(f"Audit fetch failed: {e}")

    st.divider()
    st.subheader("Right to be forgotten — delete a customer's data")
    del_cid = st.text_input("Customer ID to delete", key="delete_cid")
    del_reason = st.text_input("Reason", value="GDPR right to be forgotten", key="delete_reason")
    if st.button("Delete customer data", type="secondary"):
        if not del_cid.strip():
            st.warning("Customer ID required.")
        else:
            try:
                res = api_post("/gdpr/delete", {"customer_id": del_cid.strip(), "reason": del_reason})
                st.success(
                    f"Purged {res['records_purged']} prediction(s) · "
                    f"deletion logged at {res['deleted_at']}"
                )
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.subheader("Compliance report")
    if st.button("Generate report"):
        try:
            metrics = api_get("/model-metrics")
        except Exception as e:
            st.error(str(e))
        else:
            counts = metrics["audit_counts"]
            report = (
                "GDPR Compliance Report\n"
                f"Generated: {datetime.utcnow().isoformat()}Z\n\n"
                f"Predictions logged: {counts['predictions']}\n"
                f"Audit log entries:  {counts['audit_log']}\n"
                f"Deletions fulfilled: {counts['deletions']}\n\n"
                f"Demographic parity diff: {metrics['fairness']['demographic_parity_diff']:.3f}\n"
                f"Passes 4/5ths rule:      {metrics['fairness']['passes_4_5ths_rule']}\n\n"
                f"Model accuracy: {metrics['accuracy']:.3f}\n"
                f"Drift score:    {metrics['drift']['credit_score_drift']:.3f}\n"
            )
            st.code(report)
            st.download_button(
                "⬇️ Download report",
                report,
                file_name=f"compliance_report_{datetime.utcnow().date()}.txt",
            )


with tab_metrics:
    st.subheader("Model performance & drift")
    if st.button("Compute metrics"):
        try:
            m = api_get("/model-metrics")
        except Exception as e:
            st.error(str(e))
        else:
            cols = st.columns(4)
            cols[0].metric("Accuracy", f"{m['accuracy']:.3f}")
            cols[1].metric("Approval <50", f"{m['fairness']['approval_rate_under_50']:.3f}")
            cols[2].metric("Approval 50+", f"{m['fairness']['approval_rate_50_plus']:.3f}")
            cols[3].metric("Drift score", f"{m['drift']['credit_score_drift']:.3f}")
            st.json(m)
