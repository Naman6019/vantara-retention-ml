"""Shared dashboard CSS theme (extracted from the original dashboard)."""

CSS = r"""@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #0F172A;
}

/* Main background & universal high-contrast typography */
.stApp, .stApp p, .stApp span, .stApp div {
    color: #0F172A !important;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: #0F172A !important;
    font-weight: 700 !important;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #0F172A !important;
    border-right: 1px solid #1E293B;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #F1F5F9 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #CBD5E1 !important;
    font-size: 14px;
    font-weight: 500;
    padding: 6px 12px;
    border-radius: 8px;
    transition: all 0.2s ease-in-out;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background-color: #1E293B !important;
    color: #FFFFFF !important;
}

/* Top Executive Navigation Bar */
.crm-navbar {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.05);
}
.crm-brand-title {
    font-size: 20px;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.02em;
    margin: 0;
}
.crm-brand-sub {
    font-size: 13px;
    color: #64748B;
    margin: 2px 0 0 0;
}
.crm-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    background-color: #ECFDF5;
    color: #065F46;
    border: 1px solid #A7F3D0;
}
.crm-meta-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 500;
    background-color: #F1F5F9;
    color: #334155;
    border: 1px solid #E2E8F0;
}

/* Card System */
.crm-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.05);
    margin-bottom: 20px;
    transition: box-shadow 0.2s ease;
}
.crm-card:hover {
    box-shadow: 0 4px 12px -2px rgba(15, 23, 42, 0.08);
}

/* Tremor-style KPI Card */
.kpi-container {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.05);
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.kpi-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #0F172A;
    margin-bottom: 8px;
    line-height: 1.1;
}
.kpi-delta {
    font-size: 12px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

/* Status Badges */
.badge-emerald {
    background-color: #ECFDF5;
    color: #065F46;
    border: 1px solid #A7F3D0;
    border-radius: 9999px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
}
.badge-rose {
    background-color: #FEF2F2;
    color: #991B1B;
    border: 1px solid #FECACA;
    border-radius: 9999px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
}
.badge-amber {
    background-color: #FFFBEB;
    color: #92400E;
    border: 1px solid #FDE68A;
    border-radius: 9999px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
}
.badge-indigo {
    background-color: #EEF2FF;
    color: #3730A3;
    border: 1px solid #C7D2FE;
    border-radius: 9999px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
}

/* Benchmark System Card */
.bench-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px 18px;
    border-left: 4px solid #4F46E5;
    box-shadow: 0 1px 2px 0 rgba(15, 23, 42, 0.04);
    margin-bottom: 12px;
}
.bench-card-title {
    font-size: 13px;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 6px;
}
.bench-card-metrics {
    font-size: 12px;
    color: #475569;
    line-height: 1.6;
}

/* Customer 360 Dossier */
.c360-header {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    border-radius: 12px;
    padding: 24px;
    color: #FFFFFF;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.c360-avatar {
    width: 52px;
    height: 52px;
    background: #4F46E5;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    font-weight: 700;
    color: #FFFFFF;
    box-shadow: 0 0 12px rgba(79, 70, 229, 0.5);
}
.playbook-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #F59E0B;
    border-radius: 8px;
    padding: 14px 18px;
    margin-top: 14px;
}
"""
