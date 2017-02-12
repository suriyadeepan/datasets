ROOT='http://www.paulgraham.com/'
SEED_URL='http://www.paulgraham.com/articles.html'


from bs4 import BeautifulSoup
import requests


def get_soup(url):
    return BeautifulSoup(requests.get(url).content, 'lxml')

def get_links(url=SEED_URL):
    soup = get_soup(url)
    links = []
    for link in soup.find_all('a'):
        if 'href' in link.attrs:
            href = link.get('href')
            href = href if '/' in href else ROOT + href
            links.append(href)
    return links

def text(url):
    soup = get_soup(url)
    if soup.find('font'):
        return soup.find('font').text.strip()

def write(links, filename):
    with open(filename, 'w') as f:
        for i,link in enumerate(links[1:]):
            print('[{}] {}'.format(i,link))
            content = text(link)
            if content:
                f.write(content)
                f.write('\n')



if __name__ == '__main__':
    links = get_links(url=SEED_URL)
    write(links, 'paulg.txt')
