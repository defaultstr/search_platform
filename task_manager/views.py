#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'
from django.shortcuts import render_to_response
from django.http import HttpResponse, HttpResponseRedirect

from django.template import RequestContext

from user_system.utils import *
from .models import *


@require_login
def list_tasks(user, request):
    return render_to_response(
        'test.html',
        {
            'cur_user': user,
        },
        RequestContext(request)
    )


