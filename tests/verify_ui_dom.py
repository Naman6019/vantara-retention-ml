import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def verify_ui_and_dom():
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 950})

        print("Loading dashboard at http://localhost:8501...")
        page.goto("http://localhost:8501", timeout=30000)
        page.wait_for_selector('label[data-baseweb="radio"]', timeout=30000)
        time.sleep(2)

        radio_items = page.locator('label[data-baseweb="radio"]')
        print(f"Found {radio_items.count()} navigation items.")

        views = [
            ("executive_overview", "Executive Overview", "ui_1_executive_overview.png"),
            ("customer_segmentation", "Customer Segmentation", "ui_2_customer_segmentation.png"),
            ("churn_leaderboard", "Churn Risk Leaderboard", "ui_3_churn_leaderboard.png"),
            ("revenue_trends", "Revenue Trends & Forecast", "ui_4_revenue_trends.png"),
            ("customer_xai", "Customer XAI Deep Dive", "ui_5_customer_xai.png"),
            ("batch_scoring", "Batch Scoring & Export", "ui_6_batch_scoring.png"),
        ]

        for idx, (key, title, filename) in enumerate(views):
            print(f"Inspecting View [{idx + 1}/6]: {title}...")
            radio_items.nth(idx).click()
            time.sleep(2.5)

            h1s = page.locator("h1").all_inner_texts()
            h2s = page.locator("h2").all_inner_texts()
            h3s = page.locator("h3").all_inner_texts()

            metrics = []
            for m in page.locator('[data-testid="stMetric"]').all():
                try:
                    metrics.append({
                        "label": m.locator('[data-testid="stMetricLabel"]').inner_text(),
                        "value": m.locator('[data-testid="stMetricValue"]').inner_text()
                    })
                except Exception:
                    pass

            plotly_count = page.locator(".js-plotly-plot").count()
            sliders = page.locator('[data-testid="stSlider"]').count()
            selectboxes = page.locator('[data-testid="stSelectbox"]').count()
            tables = page.locator('[data-testid="stDataFrame"]').count() + page.locator("table").count()
            file_uploaders = page.locator('[data-testid="stFileUploader"]').count()
            buttons = page.locator("button").count()

            report[key] = {
                "view_title": title,
                "h1": h1s,
                "h2": h2s,
                "h3": h3s,
                "kpis": metrics,
                "plotly_visualizations": plotly_count,
                "interactive_sliders": sliders,
                "selectboxes": selectboxes,
                "data_tables": tables,
                "file_uploaders": file_uploaders,
                "buttons": buttons,
                "screenshot": f"docs/{filename}"
            }

        browser.close()

    report_path = docs_dir / "dom_verification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n--- COMPLETE DOM VERIFICATION REPORT ---")
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    verify_ui_and_dom()
