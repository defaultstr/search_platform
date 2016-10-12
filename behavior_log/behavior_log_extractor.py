#!/user/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

from extension_log.models import *
from behavior_log.models import *
from user_system.models import User
from eye_reader import EyeReader
from exp_domain_expertise.models import QuerySatisfactionLog
import behavior_log.utils as utils
import re
import urllib2 as urllib
from search_api.models import QueryLog
import json

DEBUG = True

def debug_print(x):
    if DEBUG:
        print x

class BehaviorLogExtractor(object):

    _query_pattern = re.compile('query=(.*?)(&|$)')
    _page_pattern = re.compile('page=(.*?)(&|$)')

    def __init__(self, user):
        if isinstance(user, User):
            self.user = user
        elif isinstance(user, str):
            try:
                self.user = User.objects.get(username=user)
            except DoesNotExist:
                raise ValueError('Username "%s" can not be found in database.')

        fixation_file = './eyetracking_log/%s.tsv' % self.user.username
        debug_print('Read fixation logs from %s' % fixation_file)
        self.eye_reader = EyeReader()
        self.eye_reader.open(fixation_file)
        debug_print('#fixations: %d' % len(self.eye_reader.fixation_df))

    def extract_task_session(self, task_url):
        start_log = ExtensionLog.objects.get(user=self.user, task_url=task_url, action='SEARCH_BEGIN')
        end_log = ExtensionLog.objects.get(user=self.user, task_url=task_url, action='SEARCH_END')

        click_timestamps = [l.timestamp
                            for l in ExtensionLog.objects(user=self.user, task_url=task_url, action='CLICK')]

        self.eye_reader.time_adjust(click_timestamps)
        fixations = self.eye_reader.get_fixations(start_time=start_log.timestamp, end_time=end_log.timestamp)

        debug_print('start time: %s' % EyeReader.timestamp_to_local_time_str(start_log.timestamp))
        debug_print('end time: %s' % EyeReader.timestamp_to_local_time_str(end_log.timestamp))
        debug_print('#fixations: %d' % len(fixations))

        task_session = TaskSession(
            user=self.user,
            task_url=task_url,
            start_time=start_log.timestamp,
            end_time=end_log.timestamp,
            pre_task_question_log=PreTaskQuestionLog.objects.get(user=self.user, task_url=task_url),
            post_task_question_log=PostTaskQuestionLog.objects.get(user=self.user, task_url=task_url),
            answer_score=-1,
        )

        debug_print(utils.document_to_json(task_session, indent=2))
        task_session.queries = self.extract_queries(task_url, fixations)


    def extract_landing_pages(self, task_url, task_session):
        c_tree = self._build_click_tree(task_url)
        r_tree = self._build_redirect_tree(task_url)
        url2page = {}
        for q in task_session.queries:
            for p in q.pages:
                url2page[p.url] = p



    def _build_click_tree(self, task_url):
        c_tree = {}
        c_logs = ExtensionLog.objects(user=self.user, task_url=task_url, action='CLICK')
        for l in c_logs:
            message = json.loads(l.message)
            if message.get('type') != 'anchor':
                continue
            c_tree[message.get('href')] = (l.site, l.timestamp)
        return c_tree

    def _build_redirect_tree(self, task_url):
        r_tree = {}
        logs = ExtensionLog.objects(user=self.user, task_url=task_url)
        last_end = None
        for l in logs:
            if last_end is None:
                if l.action == 'PAGE_END' and (not self._is_serp(l.site)):
                    last_end = l
            else:
                if l.action in ['CLICK', 'JUMP_OUT']:
                    last_end = None
                elif l.action == 'PAGE_START':
                    if self._is_serp(l.site):
                        last_end = None
                    elif l.timestamp - last_end.timestamp < 2000:
                        r_tree[l.site] = (last_end.site, last_end.timestamp)
                        last_end = None
        return r_tree

    @staticmethod
    def _is_serp(url):
        return url.startswith('http://10.129.248.120:8000/search_api/')

    @staticmethod
    def _get_query_from_site(site_url):
        site_url = site_url.encode('utf8')
        query_part = BehaviorLogExtractor._query_pattern.search(site_url).group(1)
        query = urllib.unquote(query_part).decode('utf8')
        return query

    @staticmethod
    def _get_page_num_from_site(site_url):
        match = BehaviorLogExtractor._page_pattern.search(site_url)
        page = 1
        if match is not None:
            page = int(match.group(1))
        return page

    def extract_queries(self, task_url):
        q_ext_logs = ExtensionLog.objects(user=self.user, task_url=task_url, action__in=['PAGE_START', 'PAGE_END'])
        q_ext_logs = [l for l in q_ext_logs if self._is_serp(l.site)]
        # skip first null serp
        q_ext_logs = q_ext_logs[2:]

        serps = []
        for l in q_ext_logs:
            if l.action == 'PAGE_START':
                message = json.loads(l.message)
                serp = SERPPage(
                    url=l.site,
                    query=BehaviorLogExtractor._get_query_from_site(l.site),
                    page_num=BehaviorLogExtractor._get_page_num_from_site(l.site),
                    start_time=l.timestamp,

                    html=message.get('html'),
                    mhtml=message.get('mhtml'),
                    visible_elements=message.get('visible_elements'),

                    viewports=[],
                    clicked_pages=[],
                )
                serps.append(serp)
            elif l.action == 'PAGE_END':
                query = BehaviorLogExtractor._get_query_from_site(l.site)
                assert(len(serps) > 0 and query == serps[-1].query)
                serps[-1].end_time = l.timestamp

        queries = []
        cur_query = ''
        for serp in serps:
            if serp.page_num == 1:
                query = Query(
                    query=serp.query,
                    start_time=serp.start_time,
                    pages=[],
                )
                cur_query = serp.query
                queries.append(query)
            assert(serp.query == cur_query)
            queries[-1].pages.append(serp)

        debug_print('#Queries: %d' % len(queries))
        debug_print('#SERPs: %d' % len(serps))

        q_sat_logs = QuerySatisfactionLog.objects.get(user=self.user, task_url=task_url)
        debug_print('#query satisfaction scores: %d' % len(q_sat_logs.satisfaction_scores))

        for idx, q in enumerate(queries):
            q.satisfaction_score = q_sat_logs.satisfaction_scores[idx]
            q.end_time = q.pages[-1].end_time
            debug_print('Query %d: %s\t Sat score: %d' % (idx+1, q.query,
                                    q_sat_logs.satisfaction_scores[idx].satisfaction_score))
            debug_print('start_at: %s\tend_at: %s' % (EyeReader.timestamp_to_local_time_str(q.start_time),
                                                EyeReader.timestamp_to_local_time_str(q.end_time)))
            for s in q.pages:
                debug_print('\tPage %d' % s.page_num)
                debug_print('\tstart_at: %s\tend_at: %s' % (EyeReader.timestamp_to_local_time_str(s.start_time),
                                                    EyeReader.timestamp_to_local_time_str(s.end_time)))

        return queries


def test():
    extractor = BehaviorLogExtractor('2015012338')
    extractor.extract_task_session('/exp_domain_expertise/2/')


