#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

from mongoengine import *
from user_system.models import *


class ExtensionLog(Document):
    user = ReferenceField(User)
    username = StringField()
    task_url = StringField()
    time = LongField()
    timestamp = LongField()
    action = StringField()
    message = StringField()
    site = StringField()


