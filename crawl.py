import argparse
from src.data_collecting_firefox import Crawler


BASE_URL = ("https://www.levels.fyi/jobs?from=subnav&jobFamilySlugs=data-scientist%2Cdata-analyst")

parser = argparse.ArgumentParser(
    prog='main.py',
    description='Scrape data science job postings',
    epilog=f'Thanks for using.'
)

parser.add_argument('-p', '--pages', help='set number of pages to scrape')
parser.add_argument('-u', '--url', help='set the base URL for scraping')

args = parser.parse_args()

PAGES = int(args.pages) if args.pages else 500
URL = args.url if args.url else BASE_URL


def main(pages, url):
    # url = 'https://www.levels.fyi/jobs/title/data-scientist'
    url = 'https://www.levels.fyi/jobs?from=subnav&jobFamilySlugs=data-scientist%2Cdata-analyst'
    Crawler(pages, url).run()


if __name__ == '__main__':
    main(pages=PAGES, url=URL)
