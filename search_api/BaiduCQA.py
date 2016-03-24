#coding=utf8
from bs4 import BeautifulSoup
import urllib
import urllib2
import re
import json

rc_url = re.compile(r'(?<=href=").+?(?=")')
rc_snippet = re.compile(r'(?<=<dd class="dd answer"><i class="i-answer-text">).+?(?=</dd>)')

def BaiduCQA(query):
    fr = file('Baidu.txt')
    for line in fr.readlines():
        l = line.strip().split('\t')
        if l[0] == query:
            fr.close()
            return l[1]
    fr.close()
    para = dict()
    para['word'] = query
    url = 'http://zhidao.baidu.com/search?' + urllib.urlencode(para)
    try:
        txt = urllib2.urlopen(url).read()
    except Exception as e:
        print e
    else:
        soup = BeautifulSoup(txt)
        serp = list()
        l = list()
        for result in soup.find_all('dl', class_='dl'):
            l.append(str(result))
        for result in soup.find_all('dl', class_='dl dl-last'):
            l.append(str(result))
        results = soup.find_all('dt', class_='dt mb-4 line')
        answers = soup.find_all('dd', class_='dd answer')
        for i in range(0, min(len(results),len(answers))):
            d = dict()
            result = str(results[i])
            answer = str(answers[i])
            line = result.split('\n')[1]
            d['url'] = rc_url.search(line).group()
            d['title'] = line[line.index('>')+1:line.index('</a>')]
            d['snippet'] = rc_snippet.search(answer).group().replace('</i>', '')
            d['html'] = l[i]
            serp.append(d)
        res = json.dumps(serp)
        fw = file('Baidu.txt', 'a')
        fw.write(query + '\t' + res + '\n')
        fw.close()
        return res

if __name__ == '__main__':
    data = json.loads(BaiduCQA('电风扇'))

    for d in data:
        print d['title']
        print d['snippet']
        print d['url']
        print d['html']
