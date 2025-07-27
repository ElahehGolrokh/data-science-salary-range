
from src.data_collecting import Crawler


PAGES = 3


def main():
    # url = 'https://www.google.com'
    url = 'https://www.levels.fyi/jobs/title/data-scientist'
    Crawler(PAGES, url).run()


if __name__ == '__main__':
    main()
