from playwright.sync_api import sync_playwright
import time

URL = "https://finchamp.streamlit.app/"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    print(f"Öffne {URL}...")
    page.goto(URL, timeout=60000)
    
    # Falls "Wake up"-Button vorhanden ist, klicken
    try:
        btn = page.get_by_text("Yes, get this app back up", timeout=10000)
        btn.click()
        print("Wake-up Button geklickt!")
        time.sleep(30)  # Warten bis App startet
    except:
        print("App war schon wach.")
    
    browser.close()
    print("Done.")