#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

from tasks import *
from random import shuffle


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
    print url
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