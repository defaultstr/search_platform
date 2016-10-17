#!/user/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

from mongoengine import *
from user_system.models import *
from exp_domain_expertise.models import *
from extension_log.models import *


class Fixation(Document):
    timestamp = LongField()
    origin_timestamp = LongField()
    x_on_screen = IntField()
    y_on_screen = IntField()
    x_on_page = IntField()
    y_on_page = IntField()
    duration = IntField()
    fixation_idx = IntField()


class MouseMovement(Document):
    timestamp = LongField()
    start_x = IntField()
    start_y = IntField()
    end_x = IntField()
    end_y = IntField()
    duration = IntField()


class Click(Document):
    x_on_page = IntField()
    y_on_page = IntField()
    timestamp = LongField()
    url = StringField()


class Hover(Document):
    x_on_page = IntField()
    y_on_page = IntField()
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
    hovers = ListField(ReferenceField(Hover))


class SERPPage(Document):
    username = StringField()
    task_url = StringField()

    url = StringField()
    query = StringField()
    page_num = IntField()
    start_time = LongField()
    end_time = LongField()

    html = StringField()
    mhtml = StringField()
    visible_elements = StringField()

    viewports = ListField(ReferenceField(ViewPort))
    clicked_pages = ListField(GenericReferenceField())


class LandingPage(Document):
    username = StringField()
    task_url = StringField()

    url = StringField()
    start_time = LongField()
    end_time = LongField()

    html = StringField()
    mhtml = StringField()
    visible_elements = StringField()

    usefulness_score = IntField()

    viewports = ListField(ReferenceField(ViewPort))
    clicked_pages = ListField(GenericReferenceField())

    redirect_to = GenericReferenceField()


class Query(Document):
    username = StringField()
    task_url = StringField()

    query = StringField()
    start_time = LongField()
    end_time = LongField()

    satisfaction_score = IntField()

    pages = ListField(ReferenceField(SERPPage))


class TaskSession(Document):
    user = ReferenceField(User)
    task_url = StringField()
    start_time = LongField()
    end_time = LongField()

    pre_task_question_log = ReferenceField(PreTaskQuestionLog)
    post_task_question_log = ReferenceField(PostTaskQuestionLog)
    answer_score = IntField()

    queries = ListField(ReferenceField(Query))


