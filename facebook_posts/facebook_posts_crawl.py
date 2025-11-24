from selenium.common.exceptions import TimeoutException
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from bs4 import BeautifulSoup
import re
from difflib import SequenceMatcher
import dotenv
import os

def textCleaning(text):
    text = re.sub(r'(?:facebook[\s·•▪►▶◆■—–\-]*){2,}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(See more|查看更多|顯示更多|See Translation|翻譯|Translated|已編輯|Edited|在 Facebook 上查看|在 Facebook 上查看更多)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(Like|Liked|讚|追蹤|原始音訊|留言|回應|分享|Share|Comment|Comments)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(Like|Liked|讚|追蹤|原始音訊|留言|回應|分享|Share|Comment|Comments)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[·•▪►▶◆■—–\-]+', ' ', text, flags=re.IGNORECASE)
    mixed_case_digit_re = re.compile(r'\S*[A-Z]\S*[a-z]\S*\d\S*|\S*[A-Z]\S*\d\S*[a-z]\S*|\S*[a-z]\S*[A-Z]\S*\d\S*|\S*[a-z]\S*\d\S*[A-Z]\S*|\S*\d\S*[A-Z]\S*[a-z]\S*|\S*\d\S*[a-z]\S*[A-Z]\S*', flags=re.UNICODE)
    text = mixed_case_digit_re.sub('', text)
    return text

def login_facebook(driver, email, password):
    """
    導航至 Facebook 登入頁面並執行登入操作。
    """
    print("--- 執行 Facebook 登入流程 ---")
    driver.get("https://www.facebook.com/")
    
    # 設置等待，最多等待 10 秒
    wait = WebDriverWait(driver, 10)

    try:
        # 1. 定位並輸入電子郵件/電話號碼
        # Facebook 登入框的 ID 經常是 'email' 或 'm_login_email'
        email_field = wait.until(
            EC.presence_of_element_located((By.NAME, "email"))
        )
        email_field.send_keys(email)
        print("已輸入 Email/電話號碼。")

        # 2. 定位並輸入密碼
        # Facebook 密碼框的 ID 經常是 'pass'
        password_field = wait.until(
            EC.presence_of_element_located((By.NAME, "pass"))
        )
        password_field.send_keys(password)
        print("已輸入密碼。")

        # 3. 定位並點擊登入按鈕
        # Facebook 登入按鈕的 NAME 經常是 'login'
        login_button = wait.until(
            EC.element_to_be_clickable((By.NAME, "login"))
        )
        login_button.click()
        print("點擊登入按鈕...")

        # 4. 驗證是否登入成功（等待首頁元素出現或 URL 改變）
        wait.until(EC.url_to_be("https://www.facebook.com/")) # 假設登入成功會跳轉到首頁
        
        # 登入成功後給予足夠時間載入
        time.sleep(3) 
        print("✅ 登入成功！")
        return True
    
    except Exception as e:
        print(f"❌ 登入失敗：無法找到元素或等待超時。錯誤訊息: {e}")
        # 失敗時印出當前網頁原始碼，可能有助於除錯
        # print("當前頁面原始碼:")
        # print(driver.page_source[:500])
        return False

def facebookCrawler(driver, max_scrolls=1, pause_time=1.2):
    driver.get("https://www.facebook.com/groups/1755802521338754")
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause_time)
        print(f"Scrolled to bottom {_ + 1} times.")
        # new_height = driver.execute_script("return document.body.scrollHeight")
        # if new_height == last_height:
        #     break
        # last_height = new_height
    # print(soup.prettify)
   
    open_more_xpath = "//div[contains(text(), '查看更多') or contains(text(), 'See more') or contains(text(), '展開') or contains(text(), '顯示更多')]"
    
    attempts = 0
    max_attempts = 30 # 限制點擊次數，避免無限迴圈
    wait = WebDriverWait(driver, 10)
    while attempts < max_attempts:
        try:
            # 查找所有匹配的按鈕
            open_more_buttons = driver.find_elements(By.XPATH, open_more_xpath)
            
            if not open_more_buttons:
                print(f"✅ 在第 {attempts} 次嘗試後，未找到剩餘的展開按鈕，結束展開。")
                break
                
            # 點擊找到的第一個按鈕
            driver.execute_script("arguments[0].click();", open_more_buttons[0])
            time.sleep(0.5) 
            attempts += 1
            
        except Exception as e:
            # 捕獲點擊失敗（例如：按鈕過期）
            print(f"點擊按鈕時發生錯誤：{e}，終止展開迴圈。")
            time.sleep(1) # 暫停並再試一次
            attempts += 1
            
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    output_path = "facebook_page_prettified.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(soup.prettify())
    print(f"Saved prettified HTML to {output_path}")
    results = set()
    for div in soup.find_all("div", dir="auto"):
        text = div.get_text(strip=True)
        text = textCleaning(text)
        if len(text) > 50:
            is_subset_of_existing = False
            for existing_text in results:
                if text in existing_text:
                    is_subset_of_existing = True
                    break 
            # 如果 text 不是任何現有元素的子字串，則可以將其加入
            if not is_subset_of_existing:
                results.add(text)
    for div in soup.select("div[class*='html-div']"):
        text = div.get_text(strip=True)
        text = textCleaning(text)
        
        if len(text) > 50 and len(text)<3000 and re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', text):
            is_subset_of_existing = False
            for existing_text in results:
                if text in existing_text:
                    is_subset_of_existing = True
                    break 
            # 如果 text 不是任何現有元素的子字串，則可以將其加入
            if not is_subset_of_existing:
                results.add(text)
    return results

if __name__ == "__main__":
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-images")
    chrome_options.add_argument("--disable-javascript")
    prefs = {
        # 設定 'notifications' 為 2 (Block)，即阻止所有網站的通知請求
        "profile.default_content_setting_values.notifications": 2 
    }
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=chrome_options)
    
    dotenv.load_dotenv()
    email=os.getenv("FACEBOOK_ACCOUNT")
    password=os.getenv("FACEBOOK_PASSWORD")
    articles = []
    
    try:
        if login_facebook(driver,email,password):
            articleList = facebookCrawler(driver, max_scrolls=50, pause_time=0.4)
            for text in articleList:
                articles.append(text)
            print(f"總共擷取到 {len(articles)} 篇文章。")
            articleList = facebookCrawler(driver, max_scrolls=50, pause_time=0.4)
            for text in articleList:
                articles.append(text)
            print(f"總共擷取到 {len(articles)} 篇文章。")
            articleList = facebookCrawler(driver, max_scrolls=50, pause_time=0.4)
            for text in articleList:
                articles.append(text)
            print(f"總共擷取到 {len(articles)} 篇文章。")
            articleList = facebookCrawler(driver, max_scrolls=50, pause_time=0.4)
            for text in articleList:
                articles.append(text)
            print(f"總共擷取到 {len(articles)} 篇文章。")
            
            filtered_articles = []
            for text in articles:
                should_add = True
                for existing in filtered_articles:
                    if SequenceMatcher(None, text, existing).ratio() > 0.5:
                        if len(text) < len(existing):
                            should_add = False
                        else:
                            filtered_articles.remove(existing)
                        break
                if should_add:
                    filtered_articles.append(text)
            
            print(f"經過相似度過濾後，剩下 {len(filtered_articles)} 篇文章。")
            for text in filtered_articles:
                print("-----")
                print(text)
    except Exception as e:
        print(f"An error occurred: {e}") 