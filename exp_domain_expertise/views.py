#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'
from django.shortcuts import render_to_response
from django.http import HttpResponseRedirect
from django.template import RequestContext
from user_system.utils import require_login
import task_manager.utils as task_utils
import utils
import tasks
from .models import *


@require_login
def start(user, request, task_id):
    url = utils.get_url(task_id)
    task_utils.start_task(user, url)

    task_state = None
    try:
        task_state = TaskState.objects.get(user=user, url=url)
    except DoesNotExist:
        task_state = TaskState()
    task_state.user = user
    task_state.url = url
    task_state.current_step = tasks.task_steps[0]
    task_state.save()
    return HttpResponseRedirect(utils.concat_url(task_state.url, task_state.current_step))


@require_login
def current_step(user, request, task_id):
    url = utils.get_url(task_id)
    try:
        task_state = TaskState.objects.get(user=user, url=url)
        return HttpResponseRedirect(utils.concat_url(task_state.url, task_state.current_step))
    except DoesNotExist:
        return HttpResponseRedirect(utils.concat_url(url, 'start'))


@require_login
def next_step(user, request, task_id):
    url = utils.get_url(task_id)
    try:
        task_state = TaskState.objects.get(user=user, url=url)
        n_step = utils.get_next_step(task_state.current_step)
        if n_step is None:
            task_utils.end_task(user, url)
            return HttpResponseRedirect('/task/home/')
        else:
            task_state.current_step = n_step
            task_state.save()
            return HttpResponseRedirect(utils.concat_url(task_state.url, task_state.current_step))
    except DoesNotExist:
        return HttpResponseRedirect(utils.concat_url(url, 'start'))


@require_login
def show_description(user, request, task_id):
    url = utils.get_url(task_id)
    return render_to_response(
        'exp_test.html',
        {
            'cur_user': user,
            'task_url': url,
        },
        RequestContext(request),
    )


@require_login
def pre_task_question(user, request, task_id):
    url = utils.get_url(task_id)
    return render_to_response(
        'exp_test.html',
        {
            'cur_user': user,
            'task_url': url,
        },
        RequestContext(request),
    )


@require_login
def search(user, request, task_id, query=None, page=None):
    url = utils.get_url(task_id)
    return render_to_response(
        'exp_test.html',
        {
            'cur_user': user,
            'task_url': url,
        },
        RequestContext(request),
    )


@require_login
def query_satisfaction(user, request, task_id):
    url = utils.get_url(task_id)
    return render_to_response(
        'exp_test.html',
        {
            'cur_user': user,
            'task_url': url,
        },
        RequestContext(request),
    )


@require_login
def post_task_question(user, request, task_id):
    url = utils.get_url(task_id)
    return render_to_response(
        'exp_test.html',
        {
            'cur_user': user,
            'task_url': url,
        },
        RequestContext(request),
    )



