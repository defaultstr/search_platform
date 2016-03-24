#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

from mongoengine import *
from user_system.models import *


class QueryLog(Document):
    user = ReferenceField(User)
    task_url = StringField()
    query = StringField()
    search_engine = StringField()
    page = IntField()
    timestamp = LongField()


class BingResult(EmbeddedDocument):
    title = StringField()
    url = StringField()
    snippet = StringField()
    html_content = StringField()


class BingResults(Document):
    query = StringField()
    results = ListField(EmbeddedDocumentField(BingResult))




