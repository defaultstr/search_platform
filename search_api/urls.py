#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

from django.conf.urls import url, patterns

from . import views

urlpatterns = [
    url(r'^bing/$', views.bing_search),
]

