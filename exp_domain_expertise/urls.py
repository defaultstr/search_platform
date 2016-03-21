#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'
from django.conf.urls import patterns, include, url
from . import views

urlpatterns = [
    url(r'^(\d+)/start/$', views.start),
    url(r'^(\d+)/continue/$', views.current_step),
    url(r'^(\d+)/next_step/$', views.next_step),
    url(r'^(\d+)/description/$', views.show_description),
    url(r'^(\d+)/pre_task_question/$', views.pre_task_question),
    url(r'^(\d+)/search/$', views.search),
    url(r'^(\d+)/search/(.*?)/(\d{1,2})/$', views.search),
    url(r'^(\d+)/post_task_question/$', views.post_task_question),
    url(r'^(\d+)/query_satisfaction/$', views.query_satisfaction),
]
