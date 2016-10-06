#!/user/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

from mongoengine import *
from user_system.models import *
from exp_domain_expertise.models import *
from extension_log.models import *


def document_to_json(doc):
    assert(isinstance(doc, Document))
    obj = {}


class Fixation(Document):
    timestamp = LongField()
    x_on_screen = IntField()
    y_on_screen = IntField()
    x_on_page = IntField()
    y_on_page = IntField()


class MouseMovement(Document):
    timestamp = LongField()
    start_x = IntField()
    start_y = IntField()
    end_x = IntField()
    end_y = IntField()


class Click(Document):
    timestamp = LongField()
    url = StringField()


class ViewPort(Document):
    start_time = LongField()
    end_time = LongField()
    x_pos = IntField()
    y_pos = IntField()
    fixations = ListField(ReferenceField(Fixation))
    mouse_movements = ListField(ReferenceField(MouseMovement))
    clicks = ListField(ReferenceField(Click))


class SERPPage(Document):
    page_num = IntField()
    start_time = LongField()
    end_time = LongField()
    html = StringField()
    mhtml = StringField()
    visible_elements = StringField()
    viewports = ListField(ReferenceField(ViewPort))
    clicked_pages = ListField(GenericReferenceField)


class LandingPage(Document):
    start_time = LongField()
    end_time = LongField()
    html = StringField()
    mhtml = StringField()
    visible_elements = StringField()
    usefulness_score = IntField()
    viewports = ListField(ReferenceField(ViewPort))
    clicked_pages = ListField(GenericReferenceField)


class QuerySession(Document):
    query = StringField()
    pages = ReferenceField(SERPPage)
    start_time = LongField()
    satisfaction_score = IntField()


class TaskSession(Document):
    user = ReferenceField(User)
    task_url = StringField()
    start_time = LongField()
    pre_task_question_log = ReferenceField(PreTaskQuestionLog)
    post_task_question_log = ReferenceField(PostTaskQuestionLog)
    answer_score = IntField()
    query_sessions = ListField(ReferenceField(QuerySession))


