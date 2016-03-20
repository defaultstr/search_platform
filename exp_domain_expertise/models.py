#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'
from mongoengine import *
from user_system.models import *


class TaskState(Document):
    user = ReferenceField(User)
    url = StringField()
    current_step = StringField()


class PreTaskQuestionLog(Document):
    user = ReferenceField(User)
    task = StringField()
    task_id = StringField()
    task_url = StringField()
    input_description = StringField()
    knowledge_scale = IntField()
    interest_scale = IntField()
    difficulty_scale = IntField()

