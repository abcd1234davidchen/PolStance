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

def initDB():
    conn = sqlite3.connect('title.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS titles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            label INTEGER
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_url ON titles(url)')
    conn.commit()
    conn.close()

@contextmanager
def getDBConnection():
    conn = None
    try:
        conn = sqlite3.connect('title.db', timeout=10)
        conn.execute('PRAGMA journal_mode=WAL')
        yield conn
    finally:
        if conn:
            conn.close()

def force_cleanup_database():
    try:
        conn = sqlite3.connect('title.db')
        conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        conn.close()
        print("Database WAL files cleaned up")
    except Exception as e:
        print(f"Error during cleanup: {e}")

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

def addArticleUrl(url,title,max_retries=3):
    for attempt in range(max_retries):
        try:
            with db_lock:
                with getDBConnection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('INSERT OR IGNORE INTO titles (url, title, label) VALUES (?, ?, ?)', (url, title, -1))
                    if cursor.rowcount > 0:
                        print(f"Added new URL: {url} with title: {title}")
                    conn.commit()
            return True
        except sqlite3.OperationalError as e:
            print(f"Database operation failed on attempt {attempt + 1}: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {e}")
    return False

def ctiKmtCrawler(url):
    driver = get_driver()
    articleUrls = []
    try:
        driver.get(url)
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "h3.title, article, .article-item"))
            )
        except:
            time.sleep(3)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        for section in soup.find_all('h2'):
            a_tag = section.find('a', href=True)
            if a_tag:
                href = a_tag['href']
                title = a_tag.get_text(strip=True)
                if href.startswith('/news/items'):
                    href = 'https://ctinews.com' + href
                    articleUrls.append((href, title))
        return articleUrls
    except Exception as e:
        print(f"Error crawling {url}: {e}")
        return []

def concurrentCtiKmtCrawler():
    initDB()
    CtUrls = [f"https://ctinews.com/tags/國民黨?page={page}" for page in range(1, 21)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(ctiKmtCrawler, url): url for url in CtUrls}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                article_list = future.result()
                if article_list:
                    for href, title in article_list:
                        addArticleUrl(href, title)
            except Exception as e:
                print(f"Error crawling {url}: {e}")

def chinatimesCrawler(url):
    driver = get_driver()
    articleUrls = []
    try:
        driver.get(url)
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "h3.title, article, .article-item"))
            )
        except:
            time.sleep(3)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        for section in soup.find_all('h3', class_='title'):
            a_tag = section.find('a', href=True)
            if a_tag:
                href = a_tag['href']
                title = a_tag.get_text(strip=True)
                if href.startswith('/realtimenews'):
                    href = 'https://www.chinatimes.com' + href
                    articleUrls.append((href, title))
        return articleUrls
    except Exception as e:
        print(f"Error crawling {url}: {e}")
        return []

def concurrentChinatimesCrawler():
    initDB()
    CtUrls = [f"https://www.chinatimes.com/politic/total?page={page}&chdtv" for page in range(1, 11)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(chinatimesCrawler, url): url for url in CtUrls}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                article_list = future.result()
                if article_list:
                    for href, title in article_list:
                        addArticleUrl(href, title)
            except Exception as e:
                print(f"Error crawling {url}: {e}")

def theNewsLensCrawler(url):
    driver = get_driver()
    articleUrls = []
    try:
        driver.get(url)
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "h3.title, article, .article-item"))
            )
        except:
            time.sleep(3)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        for section in soup.find_all('section', class_='item-wrapper list-item'):
            if section.find('small', class_='sponsored'):
                continue
            a_tag = section.find('a', href=True, class_='multiline-ellipsis-2 text-link')
            if a_tag:
                href = a_tag['href']
                title = a_tag.get_text(strip=True)
                if href.startswith('https://www.thenewslens.com/article/'):
                    articleUrls.append((href, title))
        return articleUrls
    except Exception as e:
        print(f"Error crawling {url}: {e}")
        return []

def concurrentTNLCrawler():
    initDB()
    TNLUrls = [f"https://www.thenewslens.com/category/politics/page{page}" for page in range(2, 100)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(theNewsLensCrawler, url): url for url in TNLUrls}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                article_list = future.result()
                if article_list:
                    for href, title in article_list:
                        addArticleUrl(href, title)
            except Exception as e:
                print(f"Error crawling {url}: {e}")

def cleanDB():
    initDB()
    with getDBConnection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM titles WHERE title IS NULL OR title = ""')
        cursor.execute('SELECT id,url,title FROM titles WHERE title IS NOT NULL AND title != ""')
        titles = cursor.fetchall()
        for title_id, url, title in tqdm(titles, desc="Cleaning DB"):
            if title is None or title=="":
                continue
            if len(title.strip()) < 18:
                cursor.execute('DELETE FROM titles WHERE id = ?', (title_id,))
        conn.commit()
            
def signal_handler(sig, frame):
    print('\nCleaning up before exit...')
    cleanup_driver()
    force_cleanup_database()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    try:
        concurrentCtiKmtCrawler()
        concurrentChinatimesCrawler()
        concurrentTNLCrawler()
        cleanDB()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Cleaning up...")
        cleanup_driver()
        force_cleanup_database()