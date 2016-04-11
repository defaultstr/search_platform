#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'


task_url = 'exp_domain_expertise'

task_steps = [
    'description',
    'pre_task_question',
    'search',
    'post_task_question',
    'query_satisfaction',
]


class Task(object):
    def __init__(self,
                 task_id=None,
                 name=None,
                 description=None,
                 **kwargs):
        self.task_id = task_id
        self.name = name
        self.description = description
        for key in kwargs:
            self.__dict__[key] = kwargs[key]

tasks = [
    Task(
        task_id='0',
        name=u'政治任务1',
        description=u'美国的利益集团为了实现自己的利益，通常会采取那些策略？',
    ),
    Task(
        task_id='1',
        name=u'政治任务2',
        description=u'政治学者注意到，美国大选中党派极化的趋势日益明显，其背后的原因有什么？（极化是指政治观点从中间向两端分散，形成两个敌对的阵营。政党认同更为强烈，更为有力地拒斥另一政党。）',
    ),
    Task(
        task_id='2',
        name=u'经济任务1',
        description=u'你认为欧债危机的起源和典型相关（国家）的应对政策各是什么？',
    ),
    Task(
        task_id='3',
        name=u'经济任务2',
        description=u'你认为我国【高校科技成果转化】的转化率大约是多少？与欧美各国相比，是高还是低？转化比例高（或低）原因有哪些？',
    ),
    Task(
        task_id='4',
        name=u'医学任务1',
        description=u'目前临床上治疗肿瘤的主要方法及其各自的优缺点？',
    ),
    Task(
        task_id='5',
        name=u'医学任务2',
        description=u'3D打印对于精准医疗有哪些可能的应用？请举例说明。',
    ),
    Task(
        task_id='6',
        name=u'环境任务1',
        description=u'请问我国颗粒物污染特征有哪些？请从全国、地区层面，时间变化层面、颗粒物组成层面等角度进行分析。',
    ),
    Task(
        task_id='7',
        name=u'环境任务2',
        description=u'饮用水消毒工艺中紫外消毒不能完全取代氯消毒的原因？',
    ),
    Task(
        task_id='8',
        name=u'物理任务1',
        description=u'引力波是怎么产生的？它和电磁波有何相同之处？',
    ),
    Task(
        task_id='9',
        name=u'物理任务2',
        description=u'有一种基本粒子叫做希格斯波色子，该粒子的存在如何解释质量的起源？',
    ),
    Task(
        task_id='10',
        name=u'法律任务1',
        description=u'根据《合同法》有关规定，房屋承租人在出租人出卖房屋于第三人时可以主张“买卖不破租赁”。设例：A将自己所有的房屋出租给B居住，在租赁期间内，A将该房屋出卖给C，请问B能否向C主张继续租赁该房屋？',
    ),
    Task(
        task_id='11',
        name=u'法律任务2',
        description=u'根据法律规定，所有权人原则上可以自有处分财产，但《合伙企业法》、《公司法》等都对企业财产有特别的规定。设例：甲出资购买一艘渔船，交由乙负责经营，乙定期将部分收益打到甲制定的账户。后甲欲将渔船出卖给丙，乙可否阻止甲将该渔船出卖？',
    ),
]
