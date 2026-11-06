import sys

from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('http://localhost:8501', timeout=25000)
    page.wait_for_selector('[data-testid="stRadio"]', timeout=25000)
    html = page.locator('[data-testid="stRadio"]').inner_html()
    print('Radio HTML snippet:')
    print(html[:1200])
    browser.close()
