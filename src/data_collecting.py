from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import numpy as np
import os
import pandas as pd
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
    _save_data()
    _get_links()
    _extract_all_job_details()
    _extract_job_detail()
    _url_exists()
    """

    def __init__(self,
                 pages: int,
                 url: str) -> None:
        self.pages = pages
        self.url = url
        self.job_links = []
        self.job_data = []

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
            self.job_links = self._get_links(driver)
            # self.jobs_df = self._extract_all_job_details(driver)
            # self._save_data()
            print('job_links lenght, final version = ', len(self.job_links))

        except Exception as e:
            print('Exception:', e)
            return {}

        finally:
            driver.quit()
            print("Selenium browser closed.")
    
    def _save_data(self):
        if not os.path.exists('/data'):
            os.mkdir('data')
        self.jobs_df.to_csv('data/raw_data.csv')

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
    
    def _extract_all_job_details(self, driver):
        jobs_df = pd.DataFrame()
        for link in self.job_links:
            try:
                data = self._extract_job_detail(driver, link)
                if data:
                    jobs_df = jobs_df.append(data, ignore_index=True)
            except Exception as e:
                print(f"Failed to extract data from {link}: {e}")
        return jobs_df

    def _extract_job_detail(self, driver, url):
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

        job_title = driver.find_element(By.XPATH, '//h1').text
        seniority_levels = ['sr', 'senior', 'jr', 'junior', 'mid-level', 'intern ']
        seniority_level = next((level for level in seniority_levels if level in job_title.lower()), None)
        roles = ['lead', 'chief', 'manager', 'head']
        role = next((role for role in roles if role in job_title.lower()), None)

        subheader_info = driver.find_element(By.XPATH, '//p[contains(@class, "job-details-header_detailsRow")]').text
        # Magna International · a month ago · Lowell, MA · Lowell, Massachusetts, United States
        company = subheader_info.split(".")[0].strip()
        post_date = subheader_info.split(".")[1].strip()
        location = subheader_info.split(".")[2:].strip()
        statuses = ['hybrid', 'remote', 'on-site']
        status = next((s for s in statuses if s in subheader_info.lower()), None)

        # Wait until the div is present
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "company-details_detailsRow"))
        )
        # Find the parent div
        company_info = driver.find_element(By.CLASS_NAME, "company-details_detailsRow")
        # Find all span elements inside the div
        spans = company_info.find_elements(By.TAG_NAME, "span")

        # Assign to variables
        ownership = company_size = revenue = headquarter = None
        for i, span in enumerate(spans):
            if i == 0:
                headquarter = span.text
            elif i == 1:
                company_size = span.text
            elif i == 2:
                revenue = span.text
            elif i == 3:
                ownership = span.text
            else:
                raise Exception("Unexpected span element in company info")

        # Wait for the section to appear
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "company-details_companyDetailsContainer"))
        )

        # Locate the section
        section = driver.find_element(By.CLASS_NAME, "company-details_companyDetailsContainer")

        # Find the <p> tag inside it
        paragraph = section.find_element(By.TAG_NAME, "p")
        company_description = paragraph.text
        industry = None

        try:
            # Try to find the salary container
            salary_divs = driver.find_elements(By.CLASS_NAME, "job-details-header_compensationRow__xsXnC")
            if salary_divs:
                salary = salary_divs[0].text  # Use the first match
            else:
                salary = None  # Not available
        except:
            salary = None

        job_description = driver.find_element(By.XPATH, '//div[contains(@class, "job-details-about_plainTextDescription__")]').text
        exprience = None

        return {
            'url': url,
            'job_title': job_title,
            'seniority_level': seniority_level,
            'role': role,
            'status': status,
            'company': company,
            'location': location,
            'job_description': job_description,
            'post_date': post_date,
            'headquarter': headquarter,
            'company_description': company_description,
            'industry': industry,
            'exprience': exprience,
            'company': company,
            'location': location,
            'ownership': ownership,
            'company_size': company_size,
            'revenue': revenue,
            'salary': salary
        }


    @staticmethod
    def _url_exists(url):
        """Checks whether the url exists"""
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            return response.status_code == 200
        except:
            return False
