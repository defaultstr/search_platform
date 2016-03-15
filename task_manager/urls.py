#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'
from django.conf.urls import patterns, include, url
from . import views

urlpatterns = [
    url(r'^home/$', views.list_tasks),
]
