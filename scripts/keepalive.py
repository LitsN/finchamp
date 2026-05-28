from playwright.sync_api import sync_playwright
import time

URL = "https://finchamp.streamlit.app/"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    print(f"Öffne {URL}...")
    page.goto(URL, timeout=60000)
    
    try:
        btn = page.locator('[data-testid="wakeup-button-viewer"]')
        btn.wait_for(timeout=10000)
        btn.click()
        print("Wake-up Button geklickt!")
        time.sleep(30)
    except:
        print("App war schon wach.")
    
    browser.close()
    print("Done.")
