#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

from tasks import *
from random import shuffle
from collections import defaultdict
from .models import *


def get_task_list():
    urls = []
    url_params = []
    names = []
    shuffled_tasks = list(tasks)
    shuffle(shuffled_tasks)
    for task in shuffled_tasks:
        urls.append(task_url)
        url_params.append(task.task_id)
        names.append(task.name)
    return urls, url_params, names


def get_task_by_id(task_id):
    ret = [task for task in tasks if task.task_id == task_id]
    if len(ret) > 0:
        return ret[0]
    else:
        return None


def get_url(task_id):
    url = '/%s/%s/' % (task_url, task_id)
    return url


def concat_url(url, compoent):
    return '%s%s/' % (url, compoent)


def get_next_step(cur_step):
    idx = task_steps.index(cur_step)
    # if all steps have been finished
    if idx == len(task_steps) - 1:
        return None
    else:
        return task_steps[idx+1]


def get_search_engine_names(name):
    if name == 'bing':
        return u'必应搜索'
    elif name == 'baidu_cqa':
        return u'百度知道'
    elif name == 'zhihu':
        return u'知乎'
    else:
        return u'其他'
