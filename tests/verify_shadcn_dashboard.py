import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def verify_and_capture_shadcn():
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # 1440x950 viewport for standard desktop CRM presentation
        context = browser.new_context(viewport={"width": 1440, "height": 950})
        page = context.new_page()

        print("Navigating to http://localhost:5173/...")
        page.goto("http://localhost:5173/", timeout=30000)
        page.wait_for_selector('text="Vantara Retail"', timeout=20000)
        time.sleep(2)

        # Tab 1: Executive Cockpit (Default)
        print("Capturing Tab 1: Executive Cockpit...")
        page.screenshot(path=str(docs_dir / "shadcn_1_overview.png"), full_page=True)

        # Tab 2: Segmentation & Personas
        print("Navigating to Tab 2: Segmentation & Personas...")
        page.locator('button[role="tab"]:has-text("Segmentation & Personas")').click()
        time.sleep(1.5)
        page.screenshot(path=str(docs_dir / "shadcn_2_segmentation.png"), full_page=True)

        # Tab 3: Retention Leaderboard
        print("Navigating to Tab 3: Retention Leaderboard...")
        page.locator('button[role="tab"]:has-text("Retention Leaderboard")').click()
        time.sleep(1.5)
        page.screenshot(path=str(docs_dir / "shadcn_3_leaderboard.png"), full_page=True)

        # Tab 4: Customer 360 & XAI
        print("Navigating to Tab 4: Customer 360 & XAI...")
        page.locator('button[role="tab"]:has-text("Customer 360 & XAI")').click()
        time.sleep(1.5)
        page.screenshot(path=str(docs_dir / "shadcn_4_customer360.png"), full_page=True)

        # Tab 5: Batch Scoring & Sync
        print("Navigating to Tab 5: Batch Scoring & Sync...")
        page.locator('button[role="tab"]:has-text("Batch Scoring & Sync")').click()
        time.sleep(1.5)
        page.screenshot(path=str(docs_dir / "shadcn_5_batch.png"), full_page=True)

        # Test Dark Theme Switch
        print("Toggling Dark Mode...")
        # Look for theme switch button
        theme_toggle = page.locator('button:has-text("Toggle theme")').or_(page.locator('[aria-label="Toggle theme"]')).or_(page.locator('button:has([data-slot="theme-switch"])')).first
        if theme_toggle.count() > 0:
            theme_toggle.click()
            time.sleep(1)
            # If dropdown menu appeared, select Dark
            dark_opt = page.locator('text="Dark"').first
            if dark_opt.count() > 0:
                dark_opt.click()
                time.sleep(1)
            # Switch back to Tab 1 to see Dark Cockpit
            page.locator('button[role="tab"]:has-text("Executive Cockpit")').click()
            time.sleep(1.5)
            page.screenshot(path=str(docs_dir / "shadcn_dark_overview.png"), full_page=True)

        browser.close()
        print("Shadcn verification and captures completed successfully!")

if __name__ == "__main__":
    verify_and_capture_shadcn()
