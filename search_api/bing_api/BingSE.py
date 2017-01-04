#coding=utf8
from bs4 import BeautifulSoup
import urllib
import urllib2
import re
import json

rc_head = re.compile(r'(?<=<h2>).+?(?=</h2>)')
rc_url = re.compile(r'(?<=href=").+?(?=")')
rc_snippet = re.compile(r'(?<=<p>).+?(?=</p>)')
rc_src = re.compile(r'(?<=src=").+?(?=")')

def BingSE(query):
    para = dict()
    para['q'] = query
    para['count'] = 50
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
            for img in BeautifulSoup(result).find_all('img', class_='rms_img'):
                src = rc_src.search(str(img)).group()
                result = result.replace(src, 'http://www.bing.com' + src)
            d = dict()
            head = rc_head.search(result).group()
            d['url'] = rc_url.search(head).group()
            d['title'] = head[head.index('>')+1:head.index('</a>')]
            match = rc_snippet.search(result)
            if match:
                d['snippet'] = rc_snippet.search(result).group()
            else:
                d['snippet'] = ''
            d['html'] = result
            serp.append(d)
        res = json.dumps(serp)
        return res

def parse_Bing_serp(txt):
    soup = BeautifulSoup(txt)
    serp = list()
    for result in soup.find_all('li', class_='b_algo'):
        result = str(result)
        for img in BeautifulSoup(result).find_all('img', class_='rms_img'):
            src = rc_src.search(str(img)).group()
            result = result.replace(src, 'http://www.bing.com' + src)
        d = dict()
        head = rc_head.search(result).group()
        d['url'] = rc_url.search(head).group()
        d['title'] = head[head.index('>')+1:head.index('</a>')]
        match = rc_snippet.search(result)
        if match:
            d['snippet'] = rc_snippet.search(result).group()
        else:
            d['snippet'] = ''
        d['html'] = result
        serp.append(d)
    return serp

if __name__ == '__main__':
    data = json.loads(BingSE('凤凰'))
    for d in data:
        print d['title'], d['snippet'], d['url']
