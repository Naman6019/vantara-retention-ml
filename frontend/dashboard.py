"""Vantara CRM — API-only Streamlit dashboard.

Every view reads from the FastAPI service (plan D6). The dashboard never loads
model artifacts or parquet files directly, so it can never drift out of sync with
the served model and it works entirely inside Docker Compose.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from frontend.theme import CSS  # noqa: E402

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TTL = 60

st.set_page_config(
    page_title="Vantara CRM | Retention & Customer Intelligence Suite",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------
@st.cache_data(ttl=TTL, show_spinner=False)
def api_get(path: str, params: dict | None = None):
    try:
        resp = requests.get(f"{API_URL}{path}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        return {"__error__": str(exc)}


def api_post(path: str, json=None, files=None):
    try:
        resp = requests.post(f"{API_URL}{path}", json=json, files=files, timeout=30)
        return resp.status_code, resp.json()
    except requests.RequestException as exc:
        return 0, {"detail": str(exc)}


def service_online() -> bool:
    health = api_get("/health")
    return isinstance(health, dict) and health.get("status") == "ok"


def require_service() -> bool:
    if service_online():
        return True
    st.error(
        f"Cannot reach the Vantara API at {API_URL}. "
        "Start the stack with `docker compose up --build` or run `uvicorn api.main:app`."
    )
    return False


def apply_theme(fig):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif", color="#334155", size=12),
        margin=dict(l=30, r=30, t=50, b=30),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0"),
    )
    return fig


def risk_badge(tier: str) -> str:
    cls = {"High": "badge-rose", "Moderate": "badge-amber"}.get(tier, "badge-emerald")
    return f'<span class="{cls}">{tier}</span>'


def kpi(title: str, value: str, delta: str = "") -> str:
    return (
        f'<div class="kpi-container"><div class="kpi-title">{title}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-delta">{delta}</div></div>'
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 16px 0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="background: linear-gradient(135deg, #4F46E5, #06B6D4); padding: 8px; border-radius: 8px; color: white; font-size: 18px; font-weight: bold;">🏢</div>
                <div>
                    <div style="color: #FFFFFF; font-weight: 700; font-size: 16px;">Vantara Retail</div>
                    <div style="color: #94A3B8; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;">CRM Intelligence Suite</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 8px 0 16px 0;'>", unsafe_allow_html=True)

    menu = st.radio(
        "CRM Navigation",
        [
            "📊 Executive CRM Cockpit",
            "🎯 At-Risk Priority (Value at Risk)",
            "👥 Customer Segmentation & Personas",
            "🔍 Customer 360 & XAI",
            "📈 Revenue Trends & Forecast",
            "🩺 Model Health & Drift",
            "🎯 Campaign Planner (Budget)",
            "📁 Batch Scoring & Export",
        ],
        index=0,
    )


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------
def view_overview():
    st.markdown("## Executive CRM Cockpit")
    if not require_service():
        return
    meta = api_get("/metadata")
    at_risk = api_get("/customers/at-risk", {"limit": 500})
    segs = api_get("/segments")
    perf = api_get("/monitoring/performance")

    total = sum(s.get("n_customers", 0) for s in segs) if isinstance(segs, list) else 0
    var_total = sum(s.get("total_value_at_risk", 0) or 0 for s in segs) if isinstance(segs, list) else 0
    high_risk = [r for r in at_risk if r.get("risk_tier") == "High"] if isinstance(at_risk, list) else []

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Customers Scored", f"{total:,}", "live batch score"), unsafe_allow_html=True)
    c2.markdown(kpi("High Churn Risk", f"{len(high_risk):,}", "calibrated tiers"), unsafe_allow_html=True)
    c3.markdown(kpi("Value at Risk", f"£{var_total:,.0f}", "P(churn) x 90-day value"), unsafe_allow_html=True)
    c4.markdown(
        kpi("Test ROC-AUC", f"{meta.get('test_roc_auc', 0):.3f}", f"threshold {meta.get('decision_threshold', 0):.2f}"),
        unsafe_allow_html=True,
    )

    st.markdown("### Model Card")
    st.markdown(
        f"""
        <div class="crm-card">
            <span class="crm-status-pill">Production</span>
            <span class="crm-meta-pill">{meta.get('version','')}</span>
            <div style="margin-top:12px;color:#475569;font-size:13px;line-height:1.8;">
                <b>Churn model:</b> {meta.get('churn_model')} &nbsp;·&nbsp;
                <b>Value model:</b> {meta.get('clv_model')} &nbsp;·&nbsp;
                <b>Segmentation:</b> {meta.get('segmentation_model')}<br>
                <b>Test ROC-AUC:</b> {meta.get('test_roc_auc')} &nbsp;·&nbsp;
                <b>Test recall:</b> {meta.get('test_recall')} &nbsp;·&nbsp;
                <b>CLV R²:</b> {meta.get('test_clv_r2')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if isinstance(segs, list) and segs:
        seg_df = pd.DataFrame(segs)
        fig = px.bar(
            seg_df, x="persona", y="total_value_at_risk", color="persona",
            title="Value at Risk by Segment", labels={"total_value_at_risk": "Value at Risk (£)", "persona": ""},
        )
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    if isinstance(perf, dict):
        val, test = perf.get("validation", {}), perf.get("test", {})
        mdf = pd.DataFrame([
            {"split": "Validation (2011-06-01)", "metric": "ROC-AUC", "value": val.get("roc_auc")},
            {"split": "Validation (2011-06-01)", "metric": "Recall @ threshold", "value": val.get("recall_at_threshold")},
            {"split": "Test (2011-09-01)", "metric": "ROC-AUC", "value": test.get("roc_auc")},
            {"split": "Test (2011-09-01)", "metric": "Recall", "value": test.get("recall")},
        ])
        st.markdown("### Out-of-time Performance")
        st.dataframe(mdf.dropna(), use_container_width=True, hide_index=True)


def view_at_risk():
    st.markdown("## At-Risk Priority — Value at Risk")
    if not require_service():
        return
    segs = api_get("/segments")
    options = ["All"] + [s["persona"] for s in segs] if isinstance(segs, list) else ["All"]
    c1, c2, c3 = st.columns(3)
    persona = c1.selectbox("Segment (persona)", options)
    country = c2.selectbox("Country", ["All", "United Kingdom", "International"])
    limit = c3.slider("Rows", 10, 500, 100, step=10)

    params = {"limit": limit}
    if persona != "All":
        params["segment"] = persona
    if country != "All":
        params["country"] = country

    rows = api_get("/customers/at-risk", params)
    if not isinstance(rows, list) or not rows:
        st.info("No at-risk customers returned. Run the batch scorer first.")
        return

    df = pd.DataFrame(rows)
    df = df[["customer_id", "calibrated_probability", "predicted_value", "value_at_risk", "risk_tier", "persona", "action"]]
    df.columns = ["Customer", "P(churn)", "90-day Value (£)", "Value at Risk (£)", "Risk", "Persona", "Recommended Action"]
    st.dataframe(
        df.style.format({"P(churn)": "{:.1%}", "90-day Value (£)": "£{:,.0f}", "Value at Risk (£)": "£{:,.0f}"}),
        use_container_width=True, hide_index=True, height=520,
    )
    st.download_button("Download CSV", df.to_csv(index=False), "at_risk_priority.csv", "text/csv")


def view_segmentation():
    st.markdown("## Customer Segmentation & Personas")
    if not require_service():
        return
    segs = api_get("/segments")
    if not isinstance(segs, list) or not segs:
        st.info("No segments available yet.")
        return
    sdf = pd.DataFrame(segs)
    fig = px.scatter(
        sdf, x="avg_churn_prob", y="avg_predicted_value", size="n_customers", color="persona",
        hover_name="persona", title="Persona Map — Risk vs Value (bubble = customers)",
        labels={"avg_churn_prob": "Avg calibrated P(churn)", "avg_predicted_value": "Avg 90-day value (£)"},
    )
    st.plotly_chart(apply_theme(fig), use_container_width=True)
    st.dataframe(sdf, use_container_width=True, hide_index=True)


def view_customer_360():
    st.markdown("## Customer 360 & Explainability (SHAP + LIME)")
    if not require_service():
        return
    top = api_get("/customers/at-risk", {"limit": 25})
    default_id = top[0]["customer_id"] if isinstance(top, list) and top else 12347
    cid = st.number_input("Customer ID", value=float(default_id), step=1.0, format="%.0f")

    cust = api_get(f"/customers/{cid}")
    if not isinstance(cust, dict) or cust.get("__error__"):
        st.warning(f"Customer {cid} not found.")
        return

    st.markdown(
        f"""
        <div class="c360-header">
            <div style="display:flex; gap:16px; align-items:center;">
                <div class="c360-avatar">{str(int(cid))[:2]}</div>
                <div>
                    <div style="font-size:20px;font-weight:700;">Customer {int(cid)}</div>
                    <div style="color:#CBD5E1;font-size:13px;">{cust.get('persona','—')}</div>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:26px;font-weight:700;">£{cust.get('value_at_risk',0):,.0f}</div>
                <div style="color:#CBD5E1;font-size:12px;">Value at Risk</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Calibrated P(churn)", f"{cust.get('calibrated_probability',0):.1%}")
    m2.metric("90-day Value", f"£{cust.get('predicted_value',0):,.0f}")
    m3.metric("Risk Tier", cust.get("risk_tier", "—"))
    m4.metric("Value Tier", (cust.get("value_tier") or "—").title())

    st.markdown(
        f'<div class="playbook-box"><b>Recommended action:</b> {cust.get("action","—")} '
        f'<span class="crm-meta-pill">{cust.get("channel","—")}</span></div>',
        unsafe_allow_html=True,
    )

    recs = api_get(f"/customers/{cid}/recommendations")
    if isinstance(recs, list) and recs:
        st.markdown("#### Recommended next products")
        rdf = pd.DataFrame(recs).reindex(columns=["product_id", "description", "because", "source"])
        rdf.columns = ["Product", "Description", "Because they bought", "Basis"]
        st.dataframe(rdf, use_container_width=True, hide_index=True)

    explain = api_get(f"/customers/{cid}/explain")
    if isinstance(explain, dict) and explain.get("shap"):
        st.info(explain.get("narrative", ""))
        left, right = st.columns(2)
        with left:
            st.markdown("#### SHAP (local)")
            sdf = pd.DataFrame(explain["shap"])
            fig = px.bar(
                sdf.sort_values("shap_impact"), x="shap_impact", y="feature", orientation="h",
                color="direction", color_discrete_map={"Increases Churn": "#DC2626", "Decreases Churn": "#059669"},
                title="SHAP contributions",
            )
            st.plotly_chart(apply_theme(fig), use_container_width=True)
        with right:
            st.markdown("#### LIME (local)")
            ldf = pd.DataFrame(explain.get("lime", []))
            if not ldf.empty:
                fig = px.bar(
                    ldf, x="weight", y="rule", orientation="h", title="LIME weights",
                    color="weight", color_continuous_scale="RdBu_r",
                )
                st.plotly_chart(apply_theme(fig), use_container_width=True)
            st.caption("SHAP and LIME are computed by the API on the served model's stored feature vector.")


def view_revenue():
    st.markdown("## Revenue Trends & Forecast")
    if not require_service():
        return
    rows = api_get("/revenue/monthly")
    if not isinstance(rows, list) or not rows:
        st.info("Revenue series unavailable.")
        return
    df = pd.DataFrame(rows)
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["month"], y=df["revenue"], mode="lines+markers", name="Monthly revenue"))
    if len(df) >= 6:
        # Naive linear forecast overlay for the next 3 months.
        x = np.arange(len(df))
        coef = np.polyfit(x, df["revenue"], 1)
        future_x = np.arange(len(df), len(df) + 3)
        future_dates = pd.date_range(df["month"].iloc[-1], periods=4, freq="MS")[1:]
        fig.add_trace(go.Scatter(
            x=future_dates, y=np.polyval(coef, future_x), mode="lines+markers",
            name="Linear forecast", line=dict(dash="dash"),
        ))
    fig.update_layout(title="Monthly Revenue with Naive Linear Forecast Overlay")
    st.plotly_chart(apply_theme(fig), use_container_width=True)


def view_health():
    st.markdown("## Model Health & Drift")
    if not require_service():
        return
    perf = api_get("/monitoring/performance")
    drift = api_get("/monitoring/drift")

    c1, c2, c3 = st.columns(3)
    c1.metric("Test ROC-AUC", perf.get("test", {}).get("roc_auc", "—"))
    c2.metric("Test Recall", perf.get("test", {}).get("recall", "—"))
    c3.metric("Calibrated Brier", perf.get("test", {}).get("calibrated_brier", "—"))

    st.markdown("### Data Drift (PSI vs training reference)")
    if isinstance(drift, list) and drift:
        ddf = pd.DataFrame(drift)
        ddf["flag"] = np.where(ddf["drifted"] == 1, "Drifted", "Stable")
        fig = px.bar(
            ddf.sort_values("psi"), x="psi", y="feature", orientation="h", color="flag",
            color_discrete_map={"Drifted": "#DC2626", "Stable": "#059669"},
            title="Population Stability Index per feature (> 0.2 flagged)",
        )
        st.plotly_chart(apply_theme(fig), use_container_width=True)
        st.dataframe(ddf, use_container_width=True, hide_index=True)
    else:
        st.info("No drift report yet. Run the batch scorer.")

    st.markdown("### Service metrics")
    st.caption("Prometheus metrics are exposed by the API at /metrics (request count, latency histogram, errors).")


def view_campaign():
    st.markdown("## Campaign Planner — Budget-Constrained Targeting")
    if not require_service():
        return
    st.caption(
        "Given a budget and an assumed save rate, the optimiser selects the customers "
        "that maximise expected retained value minus contact cost. The save rate is an "
        "assumption (no treatment data), so results are shown across a sensitivity grid."
    )
    c1, c2 = st.columns(2)
    budget = c1.slider("Budget (£)", 1000, 50000, 5000, step=500)
    save_rate = c2.slider("Assumed save rate", 0.02, 0.30, 0.10, step=0.01)

    status, body = api_post("/campaign/plan", json={"budget": float(budget), "save_rate": float(save_rate), "limit": 500})
    if status != 200:
        st.error(body.get("detail", "Campaign planning failed"))
        return

    t = body["totals"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers Selected", f"{t['n_contacted']:,}")
    k2.metric("Budget Spent", f"£{t['spent']:,.0f}")
    k3.metric("Expected Value Retained", f"£{t['expected_value_retained']:,.0f}")
    k4.metric("£ Retained per £ Spent", f"{t['value_per_pound']:.2f}")

    pol = pd.DataFrame(body["policy_comparison"])
    fig = px.bar(
        pol, x="policy", y="value_per_pound", color="policy",
        title="Policy comparison — value retained per £ spent",
    )
    st.plotly_chart(apply_theme(fig), use_container_width=True)
    st.dataframe(pol, use_container_width=True, hide_index=True)

    st.markdown("### Save-rate sensitivity")
    grid = pd.DataFrame(body["sensitivity"])
    fig = px.line(
        grid, x="budget", y="expected_net_gain", color="save_rate", markers=True,
        title="Expected net gain vs budget (per assumed save rate)",
    )
    st.plotly_chart(apply_theme(fig), use_container_width=True)

    if body["selection"]:
        sel = pd.DataFrame(body["selection"])
        st.markdown("### Selected customers (top 100)")
        st.dataframe(sel, use_container_width=True, hide_index=True)
        st.download_button("Download campaign selection", sel.to_csv(index=False), "campaign_selection.csv", "text/csv")


def view_batch():
    st.markdown("## Batch Scoring & Export")
    if not require_service():
        return
    st.write("Upload a CSV of customer features (must include the model feature columns) to score it.")
    uploaded = st.file_uploader("Customer feature CSV", type=["csv"])
    if uploaded is not None:
        status, body = api_post("/batch-score", files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")})
        if status == 200:
            out = pd.DataFrame(body)
            st.success(f"Scored {len(out):,} customers.")
            st.dataframe(out, use_container_width=True, hide_index=True)
            st.download_button("Download scored CSV", out.to_csv(index=False), "batch_scored.csv", "text/csv")
        else:
            st.error(body.get("detail", "Scoring failed"))

    st.markdown("---")
    st.write("Or download the current value-at-risk leaderboard.")
    rows = api_get("/customers/at-risk", {"limit": 500})
    if isinstance(rows, list) and rows:
        out = pd.DataFrame(rows)
        st.download_button("Download at-risk leaderboard", out.to_csv(index=False), "at_risk.csv", "text/csv")


VIEWS = {
    "📊 Executive CRM Cockpit": view_overview,
    "🎯 At-Risk Priority (Value at Risk)": view_at_risk,
    "👥 Customer Segmentation & Personas": view_segmentation,
    "🔍 Customer 360 & XAI": view_customer_360,
    "📈 Revenue Trends & Forecast": view_revenue,
    "🩺 Model Health & Drift": view_health,
    "🎯 Campaign Planner (Budget)": view_campaign,
    "📁 Batch Scoring & Export": view_batch,
}

st.markdown(
    f'<div class="crm-navbar"><div><div class="crm-brand-title">Customer Behavior Prediction Platform</div>'
    f'<div class="crm-brand-sub">Vantara Retail Solutions — CRM Intelligence Suite</div></div>'
    f'<div><span class="crm-status-pill">● {"API Online" if service_online() else "API Offline"}</span> '
    f'<span class="crm-meta-pill">{API_URL}</span></div></div>',
    unsafe_allow_html=True,
)

VIEWS[menu]()
