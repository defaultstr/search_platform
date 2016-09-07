#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

from django.shortcuts import render_to_response
from django.http import HttpResponseRedirect
from django.template import RequestContext
from user_system.utils import require_login
import task_manager.utils as task_utils
import utils
from .models import *


@require_login
def bing_search(user, request):
    # read parameters
    query = request.GET.get('query', '')
    task_url = request.GET.get('task_url', '')
    timestamp = int(request.GET.get('timestamp', '1'))
    log = int(request.GET.get('log', '1'))
    search_begin_flag = int(request.GET.get('search_begin_flag', '0'))

    page = int(request.GET.get('page', '1'))
    page = max(page, 1)
    max_page = 5
    next_page = None

    results_html = ''
    show_pagination = False
    # if query is empty, just return a search form
    if query != '':
        # check if the query has been issued
        results = None
        try:
            results = BingResults.objects.get(query=query)
        except DoesNotExist:
            # if it has not, crawl results using search api
            results = utils.crawl_bing(query)

        start_at = (page - 1) * 10
        end_at = min(len(results.results), start_at + 10)

        max_page = -(-len(results.results) / 10)
        if page < max_page:
            next_page = page + 1

        # make serp
        for idx in range(start_at, end_at):
            results_html += results.results[idx].html_content

        # log query
        if log == 1:
            query_log = QueryLog(
                user=user,
                task_url=task_url,
                query=query,
                search_engine='bing',
                page=page,
                timestamp=timestamp,
            )
            query_log.save()

    #show pagination
    if query != '':
        show_pagination = True

    return render_to_response(
        'bing_serp.html',
        {
            'cur_user': user,
            'query': query,
            'task_url': task_url,
            'log': log,
            'results_html': results_html,
            'show_pagination': show_pagination,
            'page': page,
            'pages': range(1, max_page+1),
            'next_page': next_page,
            'search_begin_flag': search_begin_flag,
        },
        RequestContext(request),
    )


@require_login
def baidu_cqa_search(user, request):
    # read parameters
    query = request.GET.get('query', '')
    task_url = request.GET.get('task_url', '')
    timestamp = int(request.GET.get('timestamp', '1'))
    log = int(request.GET.get('log', '1'))

    results_html = ''
    show_pagination = False
    # if query is empty, just return a search form
    if query != '':
        # check if the query has been issued
        results = None
        try:
            results = BaiduCQAResults.objects.get(query=query)
        except DoesNotExist:
            # if it has not, crawl results using search api
            results = utils.crawl_baidu_cqa(query)

        start_at = 0
        end_at = min(len(results.results), start_at + 10)

        # make serp
        for idx in range(start_at, end_at):
            results_html += results.results[idx].html_content

        # log query
        if log == 1:
            log = QueryLog(
                user=user,
                task_url=task_url,
                query=query,
                search_engine='baidu_cqa',
                page=1,
                timestamp=timestamp,
            )
            log.save()

    return render_to_response(
        'baidu_cqa_serp.html',
        {
            'cur_user': user,
            'query': query,
            'task_url': task_url,
            'log': log,
            'results_html': results_html,
        },
        RequestContext(request),
    )


@require_login
def zhihu_search(user, request):
    # read parameters
    query = request.GET.get('query', '')
    task_url = request.GET.get('task_url', '')
    timestamp = int(request.GET.get('timestamp', '1'))
    log = int(request.GET.get('log', '1'))

    results_html = ''
    show_pagination = False
    # if query is empty, just return a search form
    if query != '':
        # check if the query has been issued
        results = None
        try:
            results = ZhihuResults.objects.get(query=query)
        except DoesNotExist:
            # if it has not, crawl results using search api
            results = utils.crawl_zhihu(query)

        start_at = 0
        end_at = min(len(results.results), start_at + 10)

        # make serp
        for idx in range(start_at, end_at):
            results_html += results.results[idx].html_content

        # log query
        if log == 1:
            log = QueryLog(
                user=user,
                task_url=task_url,
                query=query,
                search_engine='zhihu',
                page=1,
                timestamp=timestamp,
            )
            log.save()

    return render_to_response(
        'zhihu_serp.html',
        {
            'cur_user': user,
            'query': query,
            'task_url': task_url,
            'log': log,
            'results_html': results_html,
        },
        RequestContext(request),
    )





