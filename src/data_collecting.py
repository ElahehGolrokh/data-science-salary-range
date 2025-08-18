from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import os
import pandas as pd
import time
import requests

from .utils import save_dataframe


COLUMNS = ['url', 'job_title', 'seniority_level', 'role', 'status', 'company', 'location', 'post_date', 'headquarter', 'company_description', 'industry', 'exprience', 'ownership', 'company_size', 'revenue', 'job_description', 'salary']

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
        file_path: file path for saving data

    Public Methods
    --------------
    run()

    Private Methods
    --------------
    _get_links()
    _extract_all_job_details()
    _extract_job_detail()
    _url_exists()
    _accept_cookies()
    _close_popup()
    """

    def __init__(self, pages: int,
                 url: str,
                 file_path: str,
                 dir_path: str) -> None:
        self.pages = pages
        self.url = url
        self.file_path = file_path
        self.dir_path = dir_path
        self.job_links = []
        self.jobs_df = pd.DataFrame(columns=COLUMNS)

    def run(self) -> dict:
        """
        """
        options = Options()
        options.headless = True  # Set to False if you want to see the browser
        driver = webdriver.Firefox(options=options)  # Assumes geckodriver is in PATH

        try:
            self.job_links = self._get_links(driver)
            self._extract_all_job_details(driver)
            save_dataframe(self.jobs_df,
                           self.file_path,
                           self.dir_path)
            print('job_links length, final version = ', len(self.job_links))
        except Exception as e:
            print('Exception:', e)
            return {}
        finally:
            driver.quit()
            print("Selenium browser closed.")

    def _get_links(self, driver):
        job_links = set()
        for page in range(1, self.pages + 1):
            url = self.url if page == 1 else f'{self.url}?from=subnav&offset={(page - 1) * 5}'
            if self._url_exists(url):
                driver.get(url)
                time.sleep(3)
                if page == 1:
                    self._close_popup(driver)
                    self._accept_cookies(driver)
                try:
                    WebDriverWait(driver, 20).until(
                        EC.presence_of_element_located(
                            (By.XPATH, "//*[contains(@class,'company-jobs-preview-card_companyJobsContainer')]")
                        )
                    )
                    cards = driver.find_elements(
                        By.XPATH,
                        "//*[contains(@class,'company-jobs-preview-card_companyJobsContainer')]"
                    )
                    print(f"Page {page}: Found {len(cards)} cards")
                except:
                    print(f'No more jobs found. The last page = {page - 1}')
                    break

                for card in cards:
                    cls = card.get_attribute("class")
                    if cls.startswith("company-jobs-preview-card_companyJobsContainer") and not cls.endswith("_mcTYr"):
                        try:
                            a = card.find_element(By.TAG_NAME, "a")
                            href = a.get_attribute("href")
                            if href:
                                job_links.add(href)
                        except:
                            pass

                # Locate the scrollable container (e.g. job details pane)
                # scrollable_div = driver.find_element(
                #     By.XPATH,
                #     "//*[starts-with(@class, 'jobs-directory-body_companiesListContainer')]"
                # )
                # Scroll to bottom
                # driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div)
                # time.sleep(2)
            else:
                print(f'The last page = {page - 1}')
                break
        return list(job_links)

    def _extract_all_job_details(self, driver):
        start_time = time.time()
        for i, link in enumerate(self.job_links):
            if i % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Time spent on {i/50}th 50 iterations: {elapsed:.2f} seconds")
                start_time = time.time()  # reset timer
            try:
                data = self._extract_job_detail(driver, link)
                if data:
                    self.jobs_df = pd.concat([self.jobs_df, pd.DataFrame([data])], ignore_index=True)
            except Exception as e:
                print(f"Failed to extract data from {link}: {e}")

        print('----------------------------------')
        for col in self.jobs_df.columns:
            print(f"{col}: {self.jobs_df[col].iloc[0]}")

    def _extract_job_detail(self, driver, url):
        driver.get(url)
        time.sleep(3)

        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        job_title = driver.find_element(By.XPATH, '//h1').text
        seniority_levels = ['sr', 'senior', 'jr', 'junior', 'mid-level', 'intern']
        seniority_level = next((level for level in seniority_levels if level in job_title.lower()), None)
        roles = ['lead', 'chief', 'manager', 'head']
        role = next((role for role in roles if role in job_title.lower()), None)

        subheader_info = driver.find_element(By.XPATH, '//p[contains(@class, "job-details-header_detailsRow")]').text
        company = subheader_info.split("·")[0].strip()
        post_date = subheader_info.split("·")[1].strip()
        location = '.'.join(subheader_info.split("·")[2:]).strip()
        statuses = ['hybrid', 'remote', 'on-site']
        status = next((s for s in statuses if s in subheader_info.lower()), None)

        try:
            job_description_div = driver.find_element(
                By.XPATH,
                '//div[contains(@class, "job-details-about_plainTextDescription") or contains(@class, "job-details-about_showAll")]'
            )
            job_description = job_description_div.text
        except:
            job_description = None

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH,
            "//*[starts-with(@class, 'company-details_detailsRow')]"))
        )
        
        company_info = driver.find_element(By.XPATH,
            "//*[starts-with(@class, 'company-details_detailsRow')]")
        spans = company_info.find_elements(By.TAG_NAME, "span")

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

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH,
            "//*[starts-with(@class, 'company-details_companyDetailsContainer')]"))
        )
        section = driver.find_element(By.XPATH,
            "//*[starts-with(@class, 'company-details_companyDetailsContainer')]")
        company_description = section.find_element(By.TAG_NAME, "p").text

        try:
            salary_divs = driver.find_elements(By.CLASS_NAME, "job-details-header_compensationRow__xsXnC")
            salary = salary_divs[0].text if salary_divs else None
        except:
            salary = None

        return {
            'url': url,
            'job_title': job_title,
            'seniority_level': seniority_level,
            'role': role,
            'status': status,
            'company': company,
            'location': location,
            'post_date': post_date,
            'headquarter': headquarter,
            'company_description': company_description,
            'industry': None,
            'exprience': None,
            'ownership': ownership,
            'company_size': company_size,
            'revenue': revenue,
            'job_description': job_description,
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
    
    @staticmethod
    def _accept_cookies(driver):
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//button[text()='Accept All']"))
            ).click()
            print("Cookies accepted.")
        except:
            print("No cookies banner found.")

    @staticmethod
    def _close_popup(driver):
        try:
            close_btn = driver.find_element(By.XPATH, "//*[contains(@class, 'closeButton')]")
            close_btn.click()
            print("Popup closed.")
        except:
            print("No popup found.")
