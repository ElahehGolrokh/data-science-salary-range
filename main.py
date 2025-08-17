
from src.data_collecting_firefox import Crawler

PAGES = 500


def main():
    # url = 'https://www.levels.fyi/jobs/title/data-scientist'
    url = 'https://www.levels.fyi/jobs?from=subnav&jobFamilySlugs=data-scientist%2Cdata-analyst'
    Crawler(PAGES, url).run()


if __name__ == '__main__':
    main()
