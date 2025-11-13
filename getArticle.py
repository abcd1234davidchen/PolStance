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

import siteCrawler.chinatimes as chinatimes
import siteCrawler.ctinews as ctinews
import siteCrawler.thenewslens as thenewslens
import siteCrawler.ltnnews as ltnnews
import siteCrawler.setn as setn

thread_local = threading.local()
db_lock = threading.Lock()
_cleanup_registered = False

# --- Database Functions ---

def initDB():
    conn = sqlite3.connect('article.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articleTable (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            article TEXT,
            labelA INTEGER,
            labelB INTEGER,
            labelC INTEGER,
            label INTEGER
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_url ON articleTable(url)')
    conn.commit()
    conn.close()

@contextmanager
def getDBConnection():
    conn = None
    try:
        conn = sqlite3.connect('article.db', timeout=10)
        conn.execute('PRAGMA journal_mode=WAL')
        yield conn
    finally:
        if conn:
            conn.close()

def force_cleanup_database():
    try:
        conn = sqlite3.connect('article.db')
        conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        conn.close()
        print("Database WAL files cleaned up")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def cleanDB():
    initDB()
    with getDBConnection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM articleTable WHERE title IS NULL OR title = ""')
        cursor.execute('SELECT id,title FROM articleTable WHERE title IS NOT NULL AND title != ""')
        titles = cursor.fetchall()
        for title_id, title in tqdm(titles, desc="Cleaning DB"):
            if title is None or title=="":
                continue
            if len(title.strip()) < 18:
                cursor.execute('DELETE FROM articleTable WHERE id = ?', (title_id,))
        conn.commit()
    with getDBConnection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM articleTable WHERE article = "?"')
        cursor.execute('SELECT id,article FROM articleTable WHERE article IS NOT NULL AND article != "?"')
        articles = cursor.fetchall()
        for article_id, article in tqdm(articles, desc="Cleaning DB Articles"):
            if article is None or article=="":
                continue
            if len(article.strip()) < 50:
                cursor.execute('DELETE FROM articleTable WHERE id = ?', (article_id,))
        conn.commit()

def signal_handler(sig, frame):
    print('\nCleaning up before exit...')
    cleanup_driver()
    force_cleanup_database()
    sys.exit(0)

# --- Selenium Driver Functions ---

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

# --- Page Crawlers ---

def addTitleUrl(url,title,max_retries=3):
    for attempt in range(max_retries):
        try:
            with db_lock:
                with getDBConnection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('INSERT OR IGNORE INTO articleTable (url, title, article, labelA, labelB, labelC, label) VALUES (?, ?, ?, ?, ?, ?, ?)', (url, title, "?", -1, -1, -1, -1))
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
    
def concurrentTitleCrawler(bound=[1,80,1,80,2,600,1,11,1,500,1,500]):
    initDB()
    CtiKmtUrls = [f"https://ctinews.com/tags/國民黨?page={page}" for
                   page in range(bound[0], bound[1])]
    CtiDppUrls = [f"https://ctinews.com/tags/民進黨?page={page}" for page in range(bound[2], bound[3])]
    TnlUrls = [f"https://www.thenewslens.com/category/politics/page{page}" for page in range(bound[4], bound[5])]
    CtUrls = [f"https://www.chinatimes.com/politic/total?page={page}&chdtv" for page in range(bound[6], bound[7])]
    ltnKmtUrls = [f"https://search.ltn.com.tw/list?keyword=國民黨&page={page}" for page in range(bound[8], bound[9])]
    ltnDppUrls = [f"https://search.ltn.com.tw/list?keyword=民進黨&page={page}" for page in range(bound[10], bound[11])]
    urlFullList = CtiKmtUrls+CtiDppUrls+TnlUrls+CtUrls+ltnKmtUrls+ltnDppUrls
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(pageCrawler, url): url for url in urlFullList}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                article_list = future.result()
                if article_list:
                    for href, title in article_list:
                        addTitleUrl(href, title)
            except Exception as e:
                print(f"Error crawling {url}: {e}")

def pageCrawler(url):
    driver = get_driver()
    if "ctinews.com" in url:
        return ctinews.ctiPageCrawler(driver,url)
    elif "thenewslens.com" in url:
        return thenewslens.theNewsLensPageCrawler(driver, url)
    elif "chinatimes.com" in url:
        return chinatimes.chinatimesPageCrawler(driver,url)
    elif "ltn.com" in url:
        return ltnnews.ltnPageCrawler(driver, url)
    print("Wrong url")
    return[]

def setnPageHandler(max_scrolls=1200, pause_time=1.2):
    result = setn.setnPageCrawler(get_driver(), max_scrolls=max_scrolls, pause_time=pause_time)
    for href, title in result:
        addTitleUrl(href, title)

# --- Article Crawler ---

def addArticleContent(article_id, content, max_retries=3):
    for attempt in range(max_retries):
        try:
            with db_lock:
                with getDBConnection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('UPDATE articleTable SET article = ? WHERE id = ?', (content, article_id))
                    if cursor.rowcount > 0:
                        print(f"Added article content for article ID: {article_id}")
                    conn.commit()
            return True
        except sqlite3.OperationalError as e:
            print(f"Database operation failed on attempt {attempt + 1}: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {e}")
    return False

def readTitlesWithoutArticles():
    with getDBConnection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, url FROM articleTable WHERE article = ?', ("?",))
        rows = cursor.fetchall()
    return rows

def concurrentArticleCrawler():
    rows = readTitlesWithoutArticles()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(articleCrawler, row): row for row in rows}
        for future in concurrent.futures.as_completed(future_to_url):
            row = future_to_url[future]
            try:
                article_content = future.result()
                if article_content:
                    article_id = row[0]
                    addArticleContent(article_id, article_content)
            except Exception as e:
                print(f"Error crawling article {row[1]}: {e}")

def articleCrawler(row):
    _, url = row
    driver = get_driver()
    if "ctinews.com" in url:
        return ctinews.ctiArticleCrawler(driver, url)
    elif "thenewslens.com" in url:
        return thenewslens.theNewsLensArticleCrawler(driver, url)
    elif "chinatimes.com" in url:
        return chinatimes.chinatimesArticleCrawler(driver, url)
    elif "ltn.com" in url:
        return ltnnews.ltnArticleCrawler(driver, url)
    elif "setn.com" in url:
        return setn.setnArticleCrawler(driver, url)
    print("Wrong url")
    return None

# --- Main Execution ---

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        # Daily crawl
        # concurrentTitleCrawler([1,2,1,2,2,3,1,7,1,5,1,5])
        # setnPageHandler(max_scrolls=2, pause_time=1.0)
        concurrentTitleCrawler()
        # setnPageHandler()
        concurrentArticleCrawler()
        cleanDB()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Cleaning up...")
        cleanup_driver()
        force_cleanup_database()