#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'
from mongoengine import *
from user_system.models import *


class TaskState(Document):
    user = ReferenceField(User)
    url = StringField()
    current_step = StringField()

