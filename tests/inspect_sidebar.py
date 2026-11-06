import sys

from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    print("Loading page...")
    page.goto('http://localhost:8501', timeout=30000)

    # Wait specifically for the radio navigation or the main title
    print("Waiting for radio navigation...")
    page.wait_for_selector('[data-testid="stRadio"]', timeout=30000)
    print("Radio navigation rendered!")

    # Wait for the main page title
    page.wait_for_selector('text="Customer Behavior Prediction Platform"', timeout=30000)
    print("Main title rendered!")

    # Check radio options
    options = page.locator('[data-testid="stRadio"] [role="radiogroup"] label').all_inner_texts()
    print("Radio options available:", options)

    browser.close()
