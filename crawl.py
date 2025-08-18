import argparse

from omegaconf import OmegaConf
from src.data_collecting import Crawler


parser = argparse.ArgumentParser(
    prog='crawl.py',
    description='Scrape data science job postings',
    epilog=f'Thanks for using.'
)

parser.add_argument('-p', '--pages', help='set number of pages to scrape')
parser.add_argument('-u', '--url', help='set the base URL for scraping')
parser.add_argument('-fp', '--file_path', help='set the file path for saving data')
parser.add_argument('-dp', '--dir_path', help='set the directory path for saving data')


args = parser.parse_args()

config = OmegaConf.load('config.yaml')
PAGES = int(args.pages) if args.pages else config.pages
URL = args.url if args.url else config.url
FILE_PATH = args.file_path if args.file_path else config.paths.raw_data
DIR_PATH = args.dir_path if args.dir_path else config.paths.data_dir


def main(pages: int,
         url: str,
         file_path: str,
         dir_path: str):
    Crawler(pages, url, file_path, dir_path).run()


if __name__ == '__main__':
    main(pages=PAGES, url=URL, file_path=FILE_PATH, dir_path=DIR_PATH)
