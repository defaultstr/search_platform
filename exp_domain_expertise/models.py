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


class PostTaskQuestionLog(Document):
    user = ReferenceField(User)
    task = StringField()
    task_id = StringField()
    task_url = StringField()
    question_answer = StringField()
    knowledge_scale = IntField()
    interest_scale = IntField()
    difficulty_scale = IntField()
    satisfaction_scale = IntField()


class QuerySatisfactionScore(EmbeddedDocument):
    query_index = IntField()
    query = StringField()
    search_engine = StringField()
    satisfaction_score = IntField()


class QuerySatisfactionLog(Document):
    user = ReferenceField(User)
    task = StringField()
    task_id = StringField()
    task_url = StringField()
    satisfaction_scores = ListField(EmbeddedDocumentField(QuerySatisfactionScore))
