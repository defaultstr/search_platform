#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

from .models import *


def start_task(user, url):
    try:
        task_list = TaskList.objects.get(user=user)
        combined_urls = ['/%s/%s/' % t for t in zip(task_list.urls, task_list.url_params)]
        for idx, u in enumerate(combined_urls):
            if u == url:
                task_list.states[idx] = 1
                task_list.save()
                break
    except DoesNotExist:
        pass


def end_task(user, url):
    try:
        task_list = TaskList.objects.get(user=user)
        combined_urls = ['/%s/%s/' % t for t in zip(task_list.urls, task_list.url_params)]
        for idx, u in enumerate(combined_urls):
            if u == url:
                task_list.states[idx] = 2
                task_list.save()
                break
    except DoesNotExist:
        pass
