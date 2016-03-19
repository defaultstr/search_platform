#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'
from django.shortcuts import render_to_response
from django.template import RequestContext
from user_system.utils import require_login


@require_login
def current_step(user, request, task_id):

    return render_to_response(
        'exp_test.html',
        {
            'cur_user': user,
        },
        RequestContext(request),
    )


@require_login
def next_step(user, request, task_id):
    return render_to_response(
        'exp_test.html',
        {
            'cur_user': user,
        },
        RequestContext(request),
    )


@require_login
def show_description(user, request, task_id):
    return render_to_response(
        'exp_test.html',
        {
            'cur_user': user,
        },
        RequestContext(request),
    )


@require_login
def pre_task_question(user, request, task_id):
    return render_to_response(
        'exp_test.html',
        {
            'cur_user': user,
        },
        RequestContext(request),
    )


@require_login
def search(user, request, task_id, query, page):
    return render_to_response(
        'exp_test.html',
        {
            'cur_user': user,
        },
        RequestContext(request),
    )


@require_login
def query_satisfaction(user, request, task_id):
    return render_to_response(
        'exp_test.html',
        {
            'cur_user': user,
        },
        RequestContext(request),
    )


@require_login
def post_task_question(user, request, task_id):
    return render_to_response(
        'exp_test.html',
        {
            'cur_user': user,
        },
        RequestContext(request),
    )



