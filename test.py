from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from bs4 import BeautifulSoup

import sqlite3
import os
import signal
import sys
import concurrent.futures
import threading
import atexit
from contextlib import contextmanager
from tqdm import tqdm

thread_local = threading.local()
db_lock = threading.Lock()
_cleanup_registered = False

def get_driver():
    global _cleanup_registered
    if not hasattr(thread_local, "driver"):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-images")
        chrome_options.add_argument("--disable-javascript")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        thread_local.driver = webdriver.Chrome(options=chrome_options)
        if not _cleanup_registered:
            atexit.register(cleanup_all_drivers)
            _cleanup_registered = True
    return thread_local.driver

def cleanup_driver():
    if hasattr(thread_local, 'driver'):
        try:
            thread_local.driver.quit()
            delattr(thread_local, 'driver')
        except Exception as e:
            print(f"Error closing driver: {e}")

def cleanup_all_drivers():
    try:
        cleanup_driver()
    except Exception as e:
        print(f"Error during global cleanup: {e}")

def test_site_access():
    test_urls = [
        'https://www.thenewslens.com/article/200403',
    ]
    
    for url in test_urls:
        driver = get_driver()
        try:
            print(f"Testing: {url}")
            driver.get(url)
            
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "article"))
                )
                print(f"  - Article element found")
            except:
                print(f"  - Article element not found, waiting 3 seconds...")
                time.sleep(3)
            
            html = driver.page_source

            with open('debug_output.html', 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"  - HTML saved to debug_output.html")
            
            if "Access denied" in html or "blocked" in html.lower():
                print(f"  - 可能被封鎖")
            elif "cloudflare" in html.lower():
                print(f"  - Cloudflare protection detected")
            elif len(html) < 1000:
                print(f"  - 頁面內容太少，可能有問題")
                print(f"  - HTML length: {len(html)}")
            else:
                print(f"  - 正常訪問")
                print(f"  - HTML length: {len(html)}")
                
                soup = BeautifulSoup(html, 'html.parser')
                soup = soup.find('section', class_='item article-body default-color mb-6 mt-5')
                if soup:
                    partialArticle = [p.get_text() for p in soup.find_all('p')if p.get('class') == ['ck-section']]
                    print(f"  - Found {len(partialArticle)} paragraphs with ck-section class")
                    if partialArticle:
                        sample_text = partialArticle[0][:100]
                        print(f"  - Sample text: {sample_text}...")
                else:
                    print(f"  - Article section not found")
                    
            print("-" * 50)
                
        except Exception as e:
            print(f"{url}: Error - {e}")

if __name__ == "__main__":
    try:
        test_site_access()
    finally:
        cleanup_driver()