import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def capture_all_views():
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 950})
        page = context.new_page()

        print("1. Loading dashboard...")
        page.goto("http://localhost:8501", timeout=30000)
        page.wait_for_selector('label[data-baseweb="radio"]', timeout=30000)
        time.sleep(2)

        radio_items = page.locator('label[data-baseweb="radio"]')
        print(f"Found {radio_items.count()} navigation items.")

        views = [
            ("ui_1_executive_overview.png", "Executive Overview"),
            ("ui_2_customer_segmentation.png", "Customer Segmentation"),
            ("ui_3_churn_leaderboard.png", "Churn Risk Leaderboard"),
            ("ui_4_revenue_trends.png", "Revenue Trends"),
            ("ui_5_customer_xai.png", "Customer XAI Deep Dive"),
            ("ui_6_batch_scoring.png", "Batch Scoring & Export"),
        ]

        for idx, (filename, label) in enumerate(views):
            print(f"Switching to [{idx}] {label}...")
            radio_items.nth(idx).click()
            time.sleep(3)  # Wait for charts/tables to update
            save_path = docs_dir / filename
            page.screenshot(path=str(save_path), full_page=True)
            print(f" -> Saved {save_path}")

        browser.close()
        print("All views captured successfully!")

if __name__ == "__main__":
    capture_all_views()
