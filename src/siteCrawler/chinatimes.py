from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from bs4 import BeautifulSoup

def chinatimesPageCrawler(driver,url):
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

def chinatimesArticleCrawler(driver, url):
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
        list_url = "https://www.chinatimes.com/politic/total?page=2&chdtv"
        articles = chinatimesPageCrawler(driver, list_url)
        for article_url, title in articles[0:3]:
            print(f"Title: {title}")
            content = chinatimesArticleCrawler(driver, article_url)
            print(f"Content: {content[:200]}...")  # Print first 200 characters
        # print(chinatimesArticleCrawler(driver,"https://www.chinatimes.com/realtimenews/20251113001496-260407"))
    except Exception as e:
        print(f"An error occurred: {e}")