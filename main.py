
from src.data_collecting import Crawler


def main():
    url = 'https://www.levels.fyi/jobs/title/data-scientist'
    # url = 'https://www.google.com'
    Crawler(url).run()


if __name__ == '__main__':
    main()
