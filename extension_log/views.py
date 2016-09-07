#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

from django.shortcuts import render_to_response, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .models import *
from user_system.models import User
import json


def insertMessageToDB(log_list):
    for log in log_list:
        e_log = ExtensionLog()
        try:
            e_log.user = User.objects.get(username=log['user'])
            e_log.task_url = log['task_url']
            e_log.time = log['time']
            e_log.timestamp = log['abs_time']
            e_log.action = log['action']
            e_log.message = json.dumps(log['message'], ensure_ascii=False)
            e_log.site = log['site']
            e_log.save()
        except DoesNotExist:
            pass


@csrf_exempt
def log(request):
    log_list = json.loads(request.POST[u'mouse_info'])
    insertMessageToDB(log_list)
    for log in log_list:
        print log['action'], len(json.dumps(log['message']))
    return HttpResponse('OK')

