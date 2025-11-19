from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from bs4 import BeautifulSoup
import json
import re

def ltnPageCrawler(driver, url):
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
        
        for a in soup.find_all('a', class_='tit'):
            href = a['href']
            title = a.get_text(strip=True)
            if href.startswith('https://news.ltn.com.tw'):
                articleUrls.append((href, title))
        return articleUrls
    except Exception as e:
        print(f"Error crawling {url}: {e}")
        return []

def ltnArticleCrawler(driver, url):
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
        script = soup.find('script',type="application/ld+json")
        script = json.loads(script.string.strip())
        articleText = script.get("articleBody", "")
        articleText = re.sub(r"請繼續往下閱讀[…\.]*", "", articleText)
        articleText = re.sub(r"displayDFP\([^)]*\);\s*", "", articleText)
        
        return articleText
    except Exception as e:
        print(f"Error reading article from {url}: {e}")
        return ""

if __name__ == "__main__":
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
        list_url = "https://search.ltn.com.tw/list?keyword=國民黨&page=1"
        articles = ltnPageCrawler(driver, list_url)
        for article_url, title in articles[0:3]:
            print(f"Title: {title}")
            content = ltnArticleCrawler(driver, article_url)
            print(f"Content: {content[:200]}...")  # Print first 200 characters
    except Exception as e:
        print(f"An error occurred: {e}")
