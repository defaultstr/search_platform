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
        name=u'样例任务',
        description=u'问：你认为欧债危机的起源和典型相关国家的应对政策各是什么？',
    ),
    Task(
        task_id='1',
        description=u'问：请问我国颗粒物污染特征有哪些？请从全国、地区层面，时间变化层面、颗粒物组成层面等角度进行分析。',
    ),
    Task(
        task_id='2',
        description=u'问：饮用水消毒工艺中紫外消毒不能完全取代氯消毒的原因？',
    ),
    Task(
        task_id='3',
        description=u'问：目前临床上治疗肿瘤的主要方法及其各自的优缺点？',
    ),
    Task(
        task_id='4',
        description=u'问：3D打印对于精准医疗有哪些可能的应用？',
    ),
    Task(
        task_id='5',
        description=u'问：政治学者注意到，美国大选中党派极化的趋势日益明显，其背后的原因有什么？（极化是指政治观点从中间向两端分散，形成两个敌对的阵营。政党认同更为强烈，更为有力地拒斥另一政党。）',
    ),
    Task(
        task_id='6',
        description=u'问：美国的利益集团为了实现自己的利益，通常会采取那些策略？',
    ),
]
