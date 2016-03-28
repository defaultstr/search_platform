#coding=utf8
from bs4 import BeautifulSoup
import urllib
import urllib2
import re
import json

rc_head = re.compile(r'(?<=<h2>).+?(?=</h2>)')
rc_url = re.compile(r'(?<=href=").+?(?=")')
rc_snippet = re.compile(r'(?<=<p>).+?(?=</p>)')
rc_var = re.compile(r'(?<=var x=_ge\(\').+?(?=\')')
rc_img = re.compile(r'(?<=x.src=\').+?(?=\')')
rc_key = re.compile(r'(?<=id=").+?(?=" class="rms_img")')
rc_src = re.compile(r'(?<=class="rms_img" src=").+?(?=")')

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
        img = dict()
        for line in txt.split('\n'):
            match = rc_var.search(line)
            if match:
                img[match.group()] = rc_img.search(line).group()
        soup = BeautifulSoup(txt)
        serp = list()
        for result in soup.find_all('li', class_='b_algo'):
            result = str(result)
            match = rc_key.search(result)
            if match:
                key = match.group()
                if img.has_key(key):
                    src = rc_src.search(result).group()
                    result = result.replace(src, img[key])
            d = dict()
            head = rc_head.search(result).group()
            d['url'] = rc_url.search(head).group()
            d['title'] = head[head.index('>')+1:head.index('</a>')]
            d['snippet'] = rc_snippet.search(result).group()
            d['html'] = result
            serp.append(d)
        res = json.dumps(serp)
        '''
        fw = file('Bing.txt', 'a')
        fw.write(query + '\t' + res + '\n')
        fw.close()
        '''
        return res

if __name__ == '__main__':
    data = json.loads(BingSE('凤凰'))
    for d in data:
        print d['title'], d['snippet'], d['url']
