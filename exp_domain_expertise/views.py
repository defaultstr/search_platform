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
from .forms import *


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
    task = utils.get_task_by_id(task_id)
    try:
        task_state = TaskState.objects.get(user=user, url=url)
        task_state.current_step = 'description'
        task_state.save()
    except DoesNotExist:
        return HttpResponseRedirect(utils.concat_url(url, 'start'))

    return render_to_response(
        'description.html',
        {
            'cur_user': user,
            'task_description': task.description,
            'task_url': url,
        },
        RequestContext(request),
    )


@require_login
def pre_task_question(user, request, task_id):
    url = utils.get_url(task_id)
    task = utils.get_task_by_id(task_id)
    try:
        task_state = TaskState.objects.get(user=user, url=url)
        task_state.current_step = 'pre_task_question'
        task_state.save()
    except DoesNotExist:
        return HttpResponseRedirect(utils.concat_url(url, 'start'))

    error_message = None
    if request.method == 'POST':
        form = PreTaskQuestionForm(request.POST)
        if form.is_valid():
            print 'Pre-task questionnaire:'
            print 'User: %s, Task: %s, Task ID: %s' % (user.username, url, task_id)
            print form.cleaned_data
            log = PreTaskQuestionLog()
            log.user = user
            log.task = tasks.task_url
            log.task_url = url
            log.task_id = task_id
            log.input_description = form.cleaned_data['input_description']
            log.knowledge_scale = form.cleaned_data['knowledge_scale']
            log.interest_scale = form.cleaned_data['interest_scale']
            log.difficulty_scale = form.cleaned_data['difficulty_scale']
            log.save()
            return HttpResponseRedirect(utils.concat_url(url, 'search'))
        else:
            error_message = form.errors

    return render_to_response(
        'pre_task_question.html',
        {
            'cur_user': user,
            'task_url': url,
            'error_message': error_message,
        },
        RequestContext(request),
    )


@require_login
def search(user, request, task_id, query=None, page=None):
    url = utils.get_url(task_id)
    task = utils.get_task_by_id(task_id)
    try:
        task_state = TaskState.objects.get(user=user, url=url)
        task_state.current_step = 'search'
        task_state.save()
    except DoesNotExist:
        return HttpResponseRedirect(utils.concat_url(url, 'start'))

    url = utils.get_url(task_id)
    return render_to_response(
        'search.html',
        {
            'cur_user': user,
            'task_url': url,
        },
        RequestContext(request),
    )


@require_login
def query_satisfaction(user, request, task_id):
    url = utils.get_url(task_id)
    task = utils.get_task_by_id(task_id)
    try:
        task_state = TaskState.objects.get(user=user, url=url)
        task_state.current_step = 'query_satisfaction'
        task_state.save()
    except DoesNotExist:
        return HttpResponseRedirect(utils.concat_url(url, 'start'))

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
    task = utils.get_task_by_id(task_id)
    try:
        task_state = TaskState.objects.get(user=user, url=url)
        task_state.current_step = 'post_task_question'
        task_state.save()
    except DoesNotExist:
        return HttpResponseRedirect(utils.concat_url(url, 'start'))

    error_message = None
    if request.method == 'POST':
        form = PostTaskQuestionForm(request.POST)
        if form.is_valid():
            print 'Post-task questionnaire:'
            print 'User: %s, Task: %s, Task ID: %s' % (user.username, url, task_id)
            print form.cleaned_data
            log = PostTaskQuestionLog()
            log.user = user
            log.task = tasks.task_url
            log.task_url = url
            log.task_id = task_id
            log.question_answer = form.cleaned_data['question_answer']
            log.knowledge_scale = form.cleaned_data['knowledge_scale']
            log.interest_scale = form.cleaned_data['interest_scale']
            log.difficulty_scale = form.cleaned_data['difficulty_scale']
            log.satisfaction_scale = form.cleaned_data['satisfaction_scale']
            log.save()
            return HttpResponseRedirect(utils.concat_url(url, 'query_satisfaction'))
        else:
            error_message = form.errors

    return render_to_response(
        'post_task_question.html',
        {
            'cur_user': user,
            'task_url': url,
            'task_description': task.description,
            'error_message': error_message,
        },
        RequestContext(request),
    )


