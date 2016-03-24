#coding=utf8
from bs4 import BeautifulSoup
import urllib
import urllib2
import re
import json

rc_url = re.compile(r'(?<=href=").+?(?=")')
rc_title = re.compile(r'(?<=target="_blank">).+?(?=</a>)')
rc_snippet = re.compile(r'(?<=<div class="summary hidden-expanded">).+?(?=</div>)')

def ZhihuCQA(query):
    fr = file('Zhihu.txt')
    for line in fr.readlines():
        l = line.strip().split('\t')
        if l[0] == query:
            fr.close()
            return l[1]
    fr.close()
    para = dict()
    para['q'] = query
    url = 'https://www.zhihu.com/search?' + urllib.urlencode(para)
    try:
        txt = urllib2.urlopen(url).read()
    except Exception as e:
        print e
    else:
        soup = BeautifulSoup(txt)
        serp = list()
        for result in soup.find_all('li', class_='item clearfix'):
            d = dict()
            result = str(result)
            for href in rc_url.findall(result):
                if href.startswith('/'):
                    result = result.replace(href, 'https://www.zhihu.com'+href)
            s = BeautifulSoup(result)
            for r in s.find_all('a', class_='toggle-expand inline'):
                result = result.replace(str(r), '')
            for r in s.find_all('div', class_='actions clearfix'):
                result = result.replace(str(r), '')
            d['url'] = rc_url.search(result).group()
            d['title'] = rc_title.search(result).group()
            d['snippet'] = rc_snippet.search(result).group()
            d['html'] = result
            serp.append(d)
        res = json.dumps(serp)
        fw = file('Zhihu.txt', 'a')
        fw.write(query + '\t' + res + '\n')
        fw.close()
        return res

if __name__ == '__main__':
    data = json.loads(ZhihuCQA('电风扇'))
    for d in data:
        print d['title']
        print d['snippet']
        print d['url']
        print d['html']