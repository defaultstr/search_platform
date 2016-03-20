#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'


task_url = 'exp_domain_expertise'

task_steps = [
    'description',
    'pre_task_question',
    'search',
    'question_answer',
    'query_satisfaction',
    'post_task_question',
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
        description=u'样例任务',
    ),
    Task(
        task_id='1',
        name=u'任务1',
        description=u'问：请问我国颗粒物污染特征有哪些？请从全国、地区层面，时间变化层面、颗粒物组成层面等角度进行分析。',
    ),
]
