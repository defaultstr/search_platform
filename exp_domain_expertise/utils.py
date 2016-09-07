#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

from tasks import *
from random import shuffle
from collections import defaultdict
from .models import *
from urlparse import urlparse, urlunparse, parse_qsl
from urllib import urlencode



def get_task_list():
    urls = []
    url_params = []
    names = []
    shuffled_tasks = [tasks[0]] # add sample task
    num_domain = 3
    num_turn = 2
    domain_ind = range(0, num_domain)
    shuffle(domain_ind)
    turn_ind = range(0, num_turn)
    shuffle(turn_ind)
    print domain_ind, turn_ind

    for i in range(1, 1 + num_domain*num_turn):
        turn = (i-1) / num_domain
        domain = (i-1) % num_domain

        shuffled_tasks.append(tasks[1 + turn_ind[turn] + domain_ind[domain]*num_turn])
        shuffled_tasks[i].name = u'任务 %d' % i

    for task in shuffled_tasks:
        print task.task_id
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


def add_parameters(url, parameters):
    """Parses URL and appends parameters.

    **Args:**

    * *url:* URL string.
    * *parameters:* Dict of parameters

    *Returns str*"""
    parts = list(urlparse(url))
    parts[4] = urlencode(parse_qsl(parts[4]) + parameters.items())
    return urlunparse(parts)


def add_fragments(url, fragments):
    """Parses URL and appends fragments.

    **Args:**

    * *url:* URL string.
    * *fragments:* Dict of fragments

    *Returns str*"""
    parts = list(urlparse(url))
    parts[5] = urlencode(parse_qsl(parts[5]) + fragments.items())
    return urlunparse(parts)

