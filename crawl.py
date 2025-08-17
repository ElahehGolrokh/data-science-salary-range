import argparse

from omegaconf import OmegaConf
from data_collecting import Crawler


parser = argparse.ArgumentParser(
    prog='main.py',
    description='Scrape data science job postings',
    epilog=f'Thanks for using.'
)

parser.add_argument('-p', '--pages', help='set number of pages to scrape')
parser.add_argument('-u', '--url', help='set the base URL for scraping')
parser.add_argument('-fp', '--file_path', help='set the file path for saving data')


args = parser.parse_args()

config = OmegaConf.load('config.yaml')
PAGES = int(args.pages) if args.pages else config.pages
URL = args.url if args.url else config.url
FILE_PATH = args.file_path if args.file_path else config.paths.raw_data

def main(pages, url, file_path):
    Crawler(pages, url, file_path).run()


if __name__ == '__main__':
    main(pages=PAGES, url=URL, file_path=FILE_PATH)
