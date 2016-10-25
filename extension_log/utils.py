#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = 'defaultstr'

from user_system.models import *
from extension_log.models import *
from search_api.models import QueryLog
import json


def get_user(username):
    return User.objects.get(username=username)


def get_annotations(username):
    user = get_user(username)
    logs = ExtensionLog.objects(action='USEFULNESS_ANNOTATION', user=user)
    for l in logs:
        m = json.loads(l.message)
        print l.task_url, l.site, m['usefulness_score']


def get_queries(username):
    user = get_user(username)
    logs = QueryLog.objects(user=user)
    for l in logs:
        print l.task_url, l.query, l.page


