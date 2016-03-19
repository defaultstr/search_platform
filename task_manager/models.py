#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

from mongoengine import *
from user_system.models import User

try:
    import simplejson as json
except ImportError:
    import json


class TaskList(Document):
    user = ReferenceField(User)
    urls = ListField(StringField())
    url_params = ListField(StringField())
    names = ListField(StringField())
    # 0 for un-opened, 1 for pending, 2 for finished
    states = ListField(IntField())
