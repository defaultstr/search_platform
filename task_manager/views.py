#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'
from django.shortcuts import render_to_response
from django.http import HttpResponse, HttpResponseRedirect

from django.template import RequestContext

from user_system.utils import *
from exp_domain_expertise.utils import get_task_list
from .models import *


@require_login
def list_tasks(user, request):
    task_list = None
    try:
        task_list = TaskList.objects.get(user=user)
    except DoesNotExist:
        task_list = TaskList()
        task_list.user = user
        task_list.urls, task_list.url_params, task_list.names = get_task_list()
        task_list.states = [0] * len(task_list.urls)
        task_list.save()

    ret = []
    for idx, urls in enumerate(task_list.urls):
        task = {}
        task['url'] = '/%s/%s/' % (urls, task_list.url_params[idx])
        task['name'] = task_list.names[idx]
        task['state'] = task_list.states[idx]
        ret.append(task)

    return render_to_response(
        'task_list.html',
        {
            'cur_user': user,
            'task_list': ret,
        },
        RequestContext(request)
    )


