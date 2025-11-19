from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from bs4 import BeautifulSoup
import re

def theNewsLensPageCrawler(driver, url):
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

def theNewsLensArticleCrawler(driver, url):
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
        if articleText == "":
            partialArticle = [p.get_text() for p in soup.find_all('p')]
            articleText = "\n".join(partialArticle)
        articleText = re.sub(r'【加入關鍵評論網會員】.*?立刻點擊免費加入會員！\s*', '', articleText, flags=re.DOTALL)
        articleText = re.sub(r'\n{2,}', '\n\n', articleText).strip()
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
        list_url = "https://www.thenewslens.com/category/politics/page2"
        articles = theNewsLensPageCrawler(driver, list_url)
        for article_url, title in articles[0:3]:
            print(f"Title: {title}")
            content = theNewsLensArticleCrawler(driver, article_url)
            print(f"Content: {content[:200]}...")  # Print first 200 characters
    except Exception as e:
        print(f"An error occurred: {e}") 