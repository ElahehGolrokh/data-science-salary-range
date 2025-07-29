
from src.data_collecting_firefox import Crawler
import pandas as pd

PAGES = 2


def main():
    # url = 'https://www.google.com'
    url = 'https://www.levels.fyi/jobs/title/data-scientist'
    Crawler(PAGES, url).run()


if __name__ == '__main__':
    main()
