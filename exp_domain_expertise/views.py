#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'
from django.shortcuts import render_to_response
from django.http import HttpResponseRedirect
from django.template import RequestContext, loader
from user_system.utils import require_login
import task_manager.utils as task_utils
import utils
import tasks
from .models import *
from .forms import *
from search_api.models import *


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

            next_step = utils.get_next_step(task_state.current_step)
            if next_step is None:
                task_utils.end_task(user, url)
                return HttpResponseRedirect('/task/home/')
            else:
                return HttpResponseRedirect(utils.concat_url(url, next_step))
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
            'task_description': task.description,
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

    query_logs = QueryLog.objects(user=user, task_url=url, page=1)
    error_message = None

    if request.method == "POST":
        scores = []
        for idx, query_log in enumerate(query_logs):
            score = QuerySatisfactionScore()
            score.query_index = idx+1
            score.query = query_log.query
            score.search_engine = query_log.search_engine
            sat_scale = int(request.POST.get('satisfaction_scale_%d' % (idx+1), '-1'))

            if sat_scale != -1:
                score.satisfaction_score = sat_scale
                scores.append(score)
            else:
                error_message = u'未标注查询 %d的满意度！' % (idx+1)
                break

        if error_message is None:
            log = QuerySatisfactionLog()
            log.user = user
            log.task = tasks.task_url
            log.task_url = url
            log.task_id = task_id
            log.satisfaction_scores = scores
            log.save()

            next_step = utils.get_next_step(task_state.current_step)
            if next_step is None:
                task_utils.end_task(user, url)
                return HttpResponseRedirect('/task/home/')
            else:
                return HttpResponseRedirect(utils.concat_url(url, next_step))

    queries = []
    for idx, query_log in enumerate(query_logs):
        t = loader.get_template('query_item.html')
        c = RequestContext(
            request,
            {
                'query_idx': idx+1,
                'search_engine_name': utils.get_search_engine_names(query_log.search_engine),
                'search_engine': query_log.search_engine,
                'query': query_log.query,
            }
        )
        queries.append(t.render(c))

    return render_to_response(
        'query_satisfaction.html',
        {
            'cur_user': user,
            'task_url': url,
            'task_description': task.description,
            'queries': queries,
            'error_message': error_message,
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
            log.bing_scale = form.cleaned_data['bing_scale']
            log.baidu_scale = form.cleaned_data['baidu_scale']
            log.zhihu_scale = form.cleaned_data['zhihu_scale']
            log.knowledge_scale = form.cleaned_data['knowledge_scale']
            log.interest_scale = form.cleaned_data['interest_scale']
            log.difficulty_scale = form.cleaned_data['difficulty_scale']
            log.satisfaction_scale = form.cleaned_data['satisfaction_scale']
            log.save()

            next_step = utils.get_next_step(task_state.current_step)
            if next_step is None:
                task_utils.end_task(user, url)
                return HttpResponseRedirect('/task/home/')
            else:
                return HttpResponseRedirect(utils.concat_url(url, next_step))
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


