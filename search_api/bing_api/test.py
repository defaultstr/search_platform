#coding=utf8
import json
from BingSE import BingSE
#from BaiduCQA import BaiduCQA
#from ZhihuCQA import ZhihuCQA

data = json.loads(BingSE('电风扇'))
fr = file('BingTemplate.html')
txt = fr.read()
fw = file('test.html', 'w')
serp = ''
for d in data:
    #print d['title'], d['snippet'], d['url'], d['html']
    serp += d['html']
fw.write(txt.replace('[SERP]', serp.encode('utf8')))
fw.close()
'''
data = json.loads(BaiduCQA('dota'))
for d in data:
    print d['title'], d['snippet'], d['url']

data = json.loads(ZhihuCQA('dota'))
for d in data:
    print d['title'], d['snippet'], d['url']
'''
