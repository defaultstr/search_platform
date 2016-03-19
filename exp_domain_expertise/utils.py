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


