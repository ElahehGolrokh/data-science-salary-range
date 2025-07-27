from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import requests


class Crawler:
    """
    Crawles the url with selenium 

    ...
    Attributes
    ----------
        browser: comes from selenium:
                 >>> webdriver.Chrome()
        pages: max number of pages for seeking for jobs
        url: website url

    Public Methods
    --------------
    run()

    Private Methods
    --------------
    _get_links()
    _url_exists()
    """

    def __init__(self,
                 pages: int,
                 url: str) -> None:
        self.pages = pages
        self.url = url

    def run(self) -> dict:
        """
        Returns data dictionary which is coming from selenium browser
        """
        # options = webdriver.ChromeOptions()
        # options.binary_location = '/usr/bin/google-chrome'  # Path to your Chrome binary
        # options.add_argument('--headless=chrome')  # Run in headless mode
        # options.add_argument('--no-sandbox')
        # options.add_argument('--disable-dev-shm-usage')
        # options.add_argument('--disable-gpu')
        # options.add_argument('--disable-software-rasterizer')
        # options.add_argument('--disable-accelerated-video-decode')
        # options.add_argument('--window-size=1920,1080')
        options = webdriver.ChromeOptions()
        # options.add_argument('--headless')  # remove headless if issues show up
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        try:
            job_links = self._get_links(driver)
            print('job_links lenght, final version = ', len(job_links))

        except Exception as e:
            print('Exception:', e)
            return {}

        finally:
            driver.quit()
            print("Selenium browser closed.")
    
    def _get_links(self, driver):
        job_links = set()
        for page in range(1, self.pages + 1):
            if page == 1:
                url = self.url
            else:
                url = f'{self.url}?from=subnav&offset={(page-1)*5}'
            print('page = ', page, ' , url = ', url)
            if self._url_exists(url):
                driver.get(url)
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located(
                        (By.XPATH,
                        "//*[contains(@class,'company-jobs-preview-card_companyJobsContainer')]")
                    )
                )

                cards = driver.find_elements(
                    By.XPATH,
                    "//*[contains(@class,'company-jobs-preview-card_companyJobsContainer')]"
                )
                print(f"Page {page}: Found {len(cards)} cards")

                for card in cards:
                    cls = card.get_attribute("class")
                    if cls.startswith("company-jobs-preview-card_companyJobsContainer"):
                        if not cls.endswith("_mcTYr"):  # filter out hashed suffix if needed
                            try:
                                a = card.find_element(By.TAG_NAME, "a")
                                href = a.get_attribute("href")
                                if href:
                                    job_links.add(href)
                            except:
                                pass

                # Scroll to bottom to potentially load more content and new cards
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            else:
                print(f'The last page = {page-1}')
                break


        return list(job_links)

    @staticmethod
    def _url_exists(url):
        """Checks whether the url exists"""
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            return response.status_code == 200
        except:
            return False
