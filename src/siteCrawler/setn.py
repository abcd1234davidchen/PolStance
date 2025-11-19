from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from bs4 import BeautifulSoup

def setnPageCrawler(driver, max_scrolls=1200, pause_time=1.2):
    driver.get("https://www.setn.com/ViewAll.aspx?PageGroupID=6")
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause_time)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    results = []
    for h3 in soup.find_all('h3', class_="view-li-title"):
        for a in h3.find_all('a', href=True, class_="gt"):
            href = a['href']
            title = a.get_text(strip=True)
            href = 'https://www.setn.com' + href
            results.append((href, title))
    return results

def setnArticleCrawler(driver, url, max_retries=3, initial_wait=30):
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
        articleText = "\n".join(partialArticle[:-1])
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
        articles = setnPageCrawler(driver, max_scrolls=50, pause_time=0.4)
        for article_url, title in articles[0:3]:
            print(f"Title: {title}")
            content = setnArticleCrawler(driver, article_url)
            print(f"Content: {content[:200]}...")
    except Exception as e:
        print(f"An error occurred: {e}") 