#coding=utf8
from bs4 import BeautifulSoup
import urllib
import urllib2
import re
import json

rc_head = re.compile(r'(?<=<h2>).+?(?=</h2>)')
rc_url = re.compile(r'(?<=href=").+?(?=")')
rc_snippet = re.compile(r'(?<=<p>).+?(?=</p>)')

def BingSE(query):
    fr = file('Bing.txt')
    for line in fr.readlines():
        l = line.strip().split('\t')
        if l[0] == query:
            fr.close()
            return l[1]
    fr.close()
    para = dict()
    para['q'] = query
    para['count'] = 30
    url = 'http://www.bing.com/search?' + urllib.urlencode(para)
    try:
        txt = urllib2.urlopen(url).read()
    except Exception as e:
        print e
    else:
        soup = BeautifulSoup(txt)
        serp = list()
        for result in soup.find_all('li', class_='b_algo'):
            result = str(result)
            d = dict()
            head = rc_head.search(result).group()
            d['url'] = rc_url.search(head).group()
            d['title'] = head[head.index('>')+1:head.index('</a>')]
            d['snippet'] = rc_snippet.search(result).group()
            d['html'] = result
            serp.append(d)
        res = json.dumps(serp)
        fw = file('Bing.txt', 'a')
        fw.write(query + '\t' + res + '\n')
        fw.close()
        return res

if __name__ == '__main__':
    data = json.loads(BingSE('凤凰'))
    for d in data:
        print d['title'], d['snippet'], d['url']