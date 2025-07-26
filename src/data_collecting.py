from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


class Crawler:
    """
    Crawles the url with selenium 

    ...
    Attributes
    ----------
        browser: comes from selenium:
                 >>> webdriver.Chrome()
        url: website url

    Public Methods
    --------------
    run()
    """

    def __init__(self,
                 url: str) -> None:
        self.url = url

    def run(self) -> dict:
        """
        Returns data dictionary which is coming from selenium browser
        """
        options = webdriver.ChromeOptions()
        options.binary_location = '/usr/bin/google-chrome'  # Path to your Chrome binary
        options.add_argument('--headless=chrome')  # Run in headless mode
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-software-rasterizer')
        options.add_argument('--disable-accelerated-video-decode')
        options.add_argument('--window-size=1920,1080')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        try:
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            print('url = ', self.url)
            driver.get(self.url)
            print('Page title:', driver.title)
            elements = driver.find_elements(By.XPATH,
            "//*[starts-with(@class, 'company-jobs-preview-card_companyJobTitle')]")
            print('len elements = ', len(elements))
            # Extract and print their text content
            for i, el in enumerate(elements):
                print(f"{i}th element: {el.text}")

        except Exception as e:
            print('Exception:', e)
            return {}

        finally:
            driver.quit()
            print("Selenium browser closed.")
