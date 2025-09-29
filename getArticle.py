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
    conn = sqlite3.connect('articles.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            text TEXT,
            label_kmt INTEGER,
            label_dpp INTEGER,
            label_tpp INTEGER,
            label_neutral INTEGER
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_url ON articles(url)')
    conn.commit()
    conn.close()

@contextmanager
def getDBConnection():
    conn = None
    try:
        conn = sqlite3.connect('articles.db', timeout=10)
        conn.execute('PRAGMA journal_mode=WAL')
        yield conn
    finally:
        if conn:
            conn.close()

def force_cleanup_database():
    try:
        conn = sqlite3.connect('articles.db')
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

def addArticleUrl(url,max_retries=3):
    for attempt in range(max_retries):
        try:
            with db_lock:
                with getDBConnection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('INSERT OR IGNORE INTO articles (url, text, label_kmt, label_dpp, label_tpp, label_neutral) VALUES (?, ?, ?, ?, ?, ?)', (url, "", -1, -1, -1, -1))
                    conn.commit()
            return True
        except sqlite3.OperationalError as e:
            print(f"Database operation failed on attempt {attempt + 1}: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {e}")
    return False

def updateArticleText(url, text,max_retries=3):
    for attempt in range(max_retries):
        try:
            with db_lock:
                with getDBConnection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('UPDATE articles SET text = ? WHERE url = ?', (text, url))
                    conn.commit()
                return True
        except sqlite3.OperationalError as e:
            print(f"Database operation failed on attempt {attempt + 1}: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {e}")
    return False

def getReadingArticles():
    with getDBConnection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT url FROM articles WHERE text = "" OR text IS NULL')
            # cursor.execute('SELECT url FROM articles')
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

def chinatimesCrawler():
    initDB()
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-images")
    chrome_options.add_argument("--disable-javascript")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=chrome_options)
    try:
        for page in tqdm(range(1, 5), desc="Crawling pages"):
            url = f"https://www.chinatimes.com/politic/total?page={page}&chdtv"
            driver.get(url)
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "article"))
                )
            except:
                time.sleep(3)
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/realtimenews'):
                    full_url = 'https://www.chinatimes.com' + href
                    addArticleUrl(full_url)        
    finally:
        driver.quit()

def readChinatimesArticle(url):
    driver = get_driver()
    try:
        driver.get(url)
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "article"))
            )
        except:
            time.sleep(3)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        partialArticle = [p.get_text() for p in soup.find_all('p')]
        articleText = "\n".join(partialArticle[:-5])
        print(articleText[:200])
        return articleText
    except Exception as e:
        print(f"Error reading article from {url}: {e}")
        return ""

def theNewsLensCrawler(url):
    driver = get_driver()
    articleUrls = []
    try:
        driver.get(url)
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "article"))
            )
        except:
            time.sleep(3)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        for section in soup.find_all('section', class_='item-wrapper list-item'):
            if section.find('small', class_='sponsored'):
                continue
            a_tag = section.find('a', href=True)
            if a_tag:
                href = a_tag['href']
                if href.startswith('https://www.thenewslens.com/article/'):
                    articleUrls.append(href)
        return articleUrls
    except Exception as e:
        print(f"Error crawling {url}: {e}")
        return []

def readTheNewsLensArticle(url):
    driver = get_driver()
    try:
        driver.get(url)
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "article"))
            )
        except:
            time.sleep(3)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        soup = soup.find('section', class_='item article-body default-color mb-6 mt-5')
        partialArticle = [p.get_text() for p in soup.find_all('p')if p.get('class') == ['ck-section']]
        articleText = "\n".join(partialArticle)
        print(articleText)
        return articleText
    except Exception as e:
        print(f"Error reading article from {url}: {e}")
        return ""

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
                    for article_url in article_list:
                        addArticleUrl(article_url)
                        print(f"Added article URL: {article_url}")
            except Exception as e:
                print(f"Error crawling {url}: {e}")

def readArticles(url):
    if url.startswith("https://www.chinatimes.com/realtimenews"):
        return readChinatimesArticle(url)
    elif url.startswith("https://www.thenewslens.com/article/"):
        return readTheNewsLensArticle(url)
    else:
        print(f"Unknown URL format: {url}")
        return ""

def concurrentReadArticles():
    initDB()
    articleUrls = getReadingArticles()

    if not articleUrls:
        print("No articles to read.")
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_url = {executor.submit(readArticles, url): url for url in articleUrls}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                articles_text = future.result()
                if articles_text:
                    if updateArticleText(url, articles_text):
                        print(f"Updated article from {url}")
                else:
                    print(f"No text extracted from {url}")
            except Exception as e:
                print(f"Error fetching article from {url}: {e}")
    
def signal_handler(sig, frame):
    print('\nCleaning up before exit...')
    cleanup_driver()
    force_cleanup_database()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    try:
        # chinatimesCrawler()
        # concurrentTNLCrawler()
        concurrentReadArticles()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Cleaning up...")
        cleanup_driver()
        force_cleanup_database()