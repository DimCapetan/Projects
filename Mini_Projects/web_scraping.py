import requests
import pprint
from bs4 import BeautifulSoup

res = requests.get(f'https://news.ycombinator.com/news?p=1')
res2 = requests.get(f'https://news.ycombinator.com/news?p=2')
text = res.text
text2 = res.text
soup = BeautifulSoup(text, 'html.parser')
soup2 = BeautifulSoup(text2, 'html.parser')

links = soup.select('.titleline a')
subtext = soup.select('.subtext')
links2 = soup.select('.titleline a')
subtext2 = soup.select('.subtext')


mega_links = links + links2
mega_subtext = subtext + subtext2

def sort_stories(hnlist):
    return sorted(hnlist, key = lambda key: key['votes'], reverse = True)

def create_custom_hn(links, subtext):
    hn = []
    for link, sub in zip(links, subtext):
        title = link.get_text()
        href = link.get('href', None)
        vote = sub.select('.score')
        if vote:
            points = int(vote[0].get_text().replace('points', ''))
            if points > 99:
                hn.append({'title': title, 'link': href, 'votes': points})
    return sort_stories((hn))


pprint.pprint(create_custom_hn(mega_links, mega_subtext))