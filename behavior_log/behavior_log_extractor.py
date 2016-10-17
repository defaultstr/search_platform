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
from collections import defaultdict
import urllib as urllib
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
        self.fixations = None
        self.task_session = None
        self.q_ext_logs = None
        self.active_page = None
        self.open_pages = None
        self.possible_redirect_referrer = None
        self.cur_log_idx = 0
        self.cur_log = None
        self.cur_query = None
        self.url2referrer_page = None
        self.active_viewport = None
        self.url2open_page = None
        self.all_viewports = None
        self.cur_viewport = None
        self.num_viewports = 0

        fixation_file = './eyetracking_log/%s.tsv' % self.user.username
        debug_print('Read fixation logs from %s' % fixation_file)
        self.eye_reader = EyeReader()
        self.eye_reader.open(fixation_file)
        debug_print('#fixations: %d' % len(self.eye_reader.fixation_df))

    def extract_task_session(self, task_url):
        start_log = list(ExtensionLog.objects(user=self.user, task_url=task_url, action='SEARCH_BEGIN'))[-1]
        end_log = list(ExtensionLog.objects(user=self.user, task_url=task_url, action='SEARCH_END'))[-1]

        click_timestamps = [l.timestamp
                            for l in ExtensionLog.objects(user=self.user, task_url=task_url, action='CLICK')]

        self.eye_reader.time_adjust(click_timestamps)
        self.fixations = self.eye_reader.get_fixations(start_time=start_log.timestamp, end_time=end_log.timestamp)

        debug_print('start time: %s' % EyeReader.timestamp_to_local_time_str(start_log.timestamp))
        debug_print('end time: %s' % EyeReader.timestamp_to_local_time_str(end_log.timestamp))
        debug_print('#fixations: %d' % len(self.fixations))

        self.task_session = TaskSession(
            user=self.user,
            task_url=task_url,
            start_time=start_log.timestamp,
            end_time=end_log.timestamp,
            pre_task_question_log=list(PreTaskQuestionLog.objects(user=self.user, task_url=task_url))[-1],
            post_task_question_log=list(PostTaskQuestionLog.objects(user=self.user, task_url=task_url))[-1],
            answer_score=-1,
            queries=[],
        )

        self.parse_query_and_pages(task_url)

        self.parse_viewports(task_url)

        self.fill_viewports(task_url)

        self.num_viewports = 0
        self.clean_viewports()

        debug_print('#Viewports: %d' % self.num_viewports)

        return self.task_session
        #debug_print(utils.document_to_json(self.task_session, indent=2))

    def clean_page_viewports(self, page):
        page.viewports = [vp for vp in page.viewports
                          if (len(vp.fixations) + len(vp.clicks) + len(vp.mouse_movements) + len(vp.hovers)) > 0]
        self.num_viewports += len(page.viewports)
        for p in page.clicked_pages:
            self.clean_page_viewports(p)

    def clean_viewports(self):
        for q in self.task_session.queries:
            for serp in q.pages:
                self.clean_page_viewports(serp)

    def fill_viewports(self, task_url):

        self.q_ext_logs = list(
            ExtensionLog.objects(
                user=self.user,
                task_url=task_url,
                action__in=['CLICK', 'HOVER', 'MOUSE_MOVE']
            )
        )

        log_idx = 0
        n = len(self.q_ext_logs)
        for vp in self.all_viewports:
            while log_idx < n and self.q_ext_logs[log_idx].timestamp < vp.start_time:
                log_idx += 1
            while log_idx < n and self.q_ext_logs[log_idx].timestamp <= vp.end_time:
                l = self.q_ext_logs[log_idx]
                m = json.loads(l.message)
                if l.action == 'CLICK':
                    c = Click(
                        x_on_page=m['x'],
                        y_on_page=m['y'],
                        timestamp=l.timestamp,
                        url=m['href'],
                    )
                    vp.clicks.append(c)
                elif l.action == 'HOVER':
                    h = Hover(
                        x_on_page=m['x'],
                        y_on_page=m['y'],
                        timestamp=l.timestamp,
                        url=m['href'],
                    )
                    vp.hovers.append(h)
                elif l.action == 'MOUSE_MOVE':
                    m = MouseMovement(
                        start_x=m['from']['x'],
                        start_y=m['from']['y'],
                        end_x=m['to']['x'],
                        end_y=m['to']['y'],
                        timestamp=l.timestamp,
                        duration=m['time'],
                    )
                    vp.mouse_movements.append(m)
                log_idx += 1

        for vp in self.all_viewports:
            f_df = self.fixations[(self.fixations.timestamp >= vp.start_time) &
                                  (self.fixations.timestamp <= vp.end_time)]
            for idx, row in f_df.iterrows():
                if 0 <= row['x_on_screen'] <= 1366 and 90 <= row['y_on_screen'] <= 700:
                    f = Fixation(
                        timestamp=row.timestamp,
                        origin_timestamp=row['origin_timestamp'],
                        x_on_screen=row['x_on_screen'],
                        y_on_screen=row['y_on_screen'],
                        x_on_page=row['x_on_screen']+vp.x_pos,
                        y_on_page=row['y_on_screen']-90+vp.y_pos,
                        duration=row['duration'],
                        fixation_idx=row['fixation_idx'],
                    )
                    vp.fixations.append(f)

    def end_cur_viewport(self, l):
        if self.cur_viewport is not None:
            self.cur_viewport.end_time = l.timestamp
            self.cur_viewport = None

    def new_viewport(self, l, x_pos=0, y_pos=0):
        vp = ViewPort(
            start_time=l.timestamp,
            x_pos=x_pos,
            y_pos=y_pos,

            fixations=[],
            mouse_movement=[],
            clicks=[],
            hovers=[],
        )
        page = self.find_page_by_site_and_timestamp(l.site, l.timestamp)
        if page is not None:
            self.cur_viewport = vp
            self.all_viewports.append(vp)
            page.viewports.append(vp)
            self.active_page = page

    def find_page_by_site_and_timestamp(self, url, timestamp):
        candidates = self.url2open_page.get(url)
        if candidates is None:
            return None
        candidates = [c for c in candidates if c.start_time <= timestamp <= c.end_time]
        if len(candidates) > 0:
            return candidates[0]
        return None

    def parse_viewports(self, task_url):
        self.url2open_page = defaultdict(list)
        for p in self.open_pages:
            self.url2open_page[p.url].append(p)

        self.q_ext_logs = list(
            ExtensionLog.objects(
                user=self.user,
                task_url=task_url,
                action__in=['PAGE_START', 'SCROLL', 'PAGE_END', 'JUMP_OUT', 'JUMP_IN', ],
            )
        )
        debug_print('#viewport extension logs: %d' % len(self.q_ext_logs))
        debug_print('#viewport extension logs: %d' % len([l for l in self.q_ext_logs if l.action == 'SCROLL']))
        self.cur_log_idx = 0
        self.cur_log = None

        self.active_page = None
        self.cur_viewport = None
        self.all_viewports = []

        while self.cur_log_idx < len(self.q_ext_logs):
            if self.accept_log(accept_actions=['PAGE_START'], extra_conditions=[self.cond_is_serp]):
                self.end_cur_viewport(self.cur_log)
                self.new_viewport(self.cur_log)
            elif self.accept_log(accept_actions=['PAGE_START']):
                if self.cur_viewport is None:
                    self.new_viewport(self.cur_log)
            elif self.accept_log(accept_actions=['SCROLL']):
                l = self.cur_log
                m = json.loads(self.cur_log.message)

                self.new_viewport(self.cur_log, int(m['to']['x']), int(m['to']['y']))
            elif self.accept_log(accept_actions=['PAGE_END']):
                if self.active_page and self.active_page.url == self.cur_log.site:
                    self.end_cur_viewport(self.cur_log)
            elif self.accept_log(accept_actions=['JUMP_OUT']):
                self.end_cur_viewport(self.cur_log)
            elif self.accept_log(accept_actions=['JUMP_IN']):
                l = self.cur_log
                page = self.find_page_by_site_and_timestamp(l.site, l.timestamp)
                if page:
                    if len(page.viewports) == 0:
                        self.new_viewport(l)
                    else:
                        last_vp = page.viewports[-1]
                        self.new_viewport(l, x_pos=last_vp.x_pos, y_pos=last_vp.y_pos)
            else:
                break

    def parse_query_and_pages(self, task_url):
        self.q_ext_logs = list(
            ExtensionLog.objects(
                user=self.user,
                task_url=task_url,
                action__in=['PAGE_START', 'PAGE_END', 'JUMP_OUT', 'JUMP_IN', 'CLICK', 'USEFULNESS_ANNOTATION'],
            )
        )
        self.cur_log_idx = 0
        self.cur_log = None

        self.active_page = None
        self.cur_query = None
        self.open_pages = []
        self.url2referrer_page = {}
        self.possible_redirect_referrer = None

        self.accept_until(wait_actions=['PAGE_START'])
        self.accept_until(wait_actions=['PAGE_END'])

        while self.cur_log_idx < len(self.q_ext_logs):
            if self.accept_log(accept_actions=['PAGE_START', 'PAGE_END'], extra_conditions=[self.cond_is_search_start]):
                pass
            elif self.accept_log(accept_actions=['PAGE_START'], extra_conditions=[self.cond_is_query]):
                l = self.cur_log
                m = json.loads(l.message)

                serp = SERPPage(
                    username=self.user.username,
                    task_url=task_url,

                    url=l.site,
                    query=BehaviorLogExtractor._get_query_from_site(l.site),
                    page_num=BehaviorLogExtractor._get_page_num_from_site(l.site),
                    start_time=l.timestamp,

                    html=m.get('html'),
                    mhtml=m.get('mhtml'),

                    visible_elements=json.dumps(m.get('visible_elements'), ensure_ascii=False),

                    viewports=[],
                    clicked_pages=[],
                )

                self.active_page = serp
                self.open_pages.append(serp)
                self.cur_query = Query(
                    username=self.user.username,
                    task_url=task_url,

                    query=serp.query,
                    start_time=serp.start_time,
                    pages=[serp],
                )
                self.task_session.queries.append(self.cur_query)
            elif self.accept_log(accept_actions=['PAGE_START'], extra_conditions=[self.cond_is_serp]):
                l = self.cur_log
                m = json.loads(l.message)

                serp = SERPPage(
                    username=self.user.username,
                    task_url=task_url,

                    url=l.site,
                    query=BehaviorLogExtractor._get_query_from_site(l.site),
                    page_num=BehaviorLogExtractor._get_page_num_from_site(l.site),
                    start_time=l.timestamp,

                    html=m.get('html'),
                    mhtml=m.get('mhtml'),

                    visible_elements=json.dumps(m.get('visible_elements'), ensure_ascii=False),

                    viewports=[],
                    clicked_pages=[],
                )
                self.active_page = serp
                self.open_pages.append(serp)

                self.cur_query.pages.append(serp)

            elif self.accept_log(accept_actions=['PAGE_START']):
                l = self.cur_log
                m = json.loads(l.message)

                p = LandingPage(
                    username=self.user.username,
                    task_url=task_url,

                    url=l.site,
                    start_time=l.timestamp,

                    usefulness_score=1,

                    html=m.get('html'),
                    mhtml=m.get('mhtml'),

                    visible_elements=json.dumps(m.get('visible_elements'), ensure_ascii=False),

                    viewports=[],
                    clicked_pages=[],
                )

                self.open_pages.append(p)

                refer_page = self.url2referrer_page.get(p.url)
                if refer_page is not None:
                    refer_page.clicked_pages.append(p)
                    del self.url2referrer_page[p.url]
            elif self.accept_log(accept_actions=['PAGE_END']):
                l = self.cur_log

                closed_pages = [p for p in self.open_pages if p.url == l.site]
                if len(closed_pages) > 0:
                    closed_pages[0].end_time = l.timestamp

                    self.possible_redirect_referrer = closed_pages[0]

                    if closed_pages[0] == self.active_page:
                        self.active_page = None
            elif self.accept_log(accept_actions=['JUMP_OUT']):
                self.active_page = None
            elif self.accept_log(accept_actions=['JUMP_IN']):
                l = self.cur_log

                jump_in_pages = [p for p in self.open_pages if p.url == l.site]
                if len(jump_in_pages) > 0:
                    self.active_page = jump_in_pages[0]
            elif self.accept_log(accept_actions=['CLICK']):
                l = self.cur_log
                m = json.loads(l.message)
                if self.active_page is not None:
                    self.url2referrer_page[m['href']] = self.active_page
                else:
                    refer_pages = [p for p in self.open_pages if p.url == l.site]
                    if len(refer_pages) > 0:
                        self.url2referrer_page[m['href']] = refer_pages[0]
            elif self.accept_log(accept_actions=['USEFULNESS_ANNOTATION']):
                l = self.cur_log
                m = json.loads(l.message)
                annotated_page = self.active_page
                if annotated_page is None or (not isinstance(annotated_page, LandingPage)):
                    pages = [p for p in self.open_pages if p.url == l.site and isinstance(p, LandingPage)]
                    if len(pages) > 0:
                        annotated_page = pages[0]
                if annotated_page is not None:
                    annotated_page.usefulness_score = m['usefulness_score']
            else:
                break

        self.task_session.queries = [
            q for q in self.task_session.queries
            if max([s.end_time for s in q.pages]) is not None
        ]
        for idx, q in enumerate(self.task_session.queries):
            q.end_time = max([s.end_time for s in q.pages])

        q_sat_log = QuerySatisfactionLog.objects.get(user=self.user, task_url=task_url)
        debug_print('#query satisfaction scores: %d' % len(q_sat_log.satisfaction_scores))

        q2sat = {
            q_sat_log.satisfaction_scores[idx].query: q_sat_log.satisfaction_scores[idx].satisfaction_score
            for idx in range(len(q_sat_log.satisfaction_scores))
        }

        for idx, q in enumerate(self.task_session.queries):
            q.end_time = max([s.end_time for s in q.pages])
            q.satisfaction_score = q2sat[q.query]
            debug_print('Query %d: %s\t Sat score: %d' % (idx+1, q.query, q.satisfaction_score))
            debug_print('start_at: %s\tend_at: %s' %
                        (EyeReader.timestamp_to_local_time_str(q.start_time),
                         EyeReader.timestamp_to_local_time_str(q.end_time)))

    def cond_is_serp(self, log):
        return log.site.startswith('http://10.129.248.120:8000/search_api/')

    def cond_is_query(self, log):
        return self.cond_is_serp(log) and self._get_page_num_from_site(log.site) == 1

    def cond_is_search_start(self, log):
        return log.site.startswith('http://10.129.248.120:8000/search_api/bing/?search_begin_flag=1')

    def test_conditions(self, log, conditions):
        for c in conditions:
            if not c(log):
                return False
        return True

    def assert_log(self, assert_actions=[], extra_conditions=[]):
        if not self.accept_log(accept_actions=assert_actions, extra_conditions=extra_conditions):
            debug_print(self.q_ext_logs[self.cur_log_idx].action)
            assert self.accept_log(accept_actions=assert_actions)

    def accept_log(self, accept_actions=[], extra_conditions=[]):
        log = self.q_ext_logs[self.cur_log_idx]
        if log.action in accept_actions and self.test_conditions(log, extra_conditions):
            self.cur_log = log
            self.cur_log_idx += 1
            '''
            if self.cur_log_idx % 1000 == 0:
                print self.cur_log_idx
            '''
            return True
        else:
            return False

    def accept_until(self, wait_actions=[], extra_conditions=[]):
        log = self.q_ext_logs[self.cur_log_idx]
        while (log.action not in wait_actions) or (not self.test_conditions(log, extra_conditions)):
            self.cur_log_idx += 1
            log = self.q_ext_logs[self.cur_log_idx]
        self.cur_log = log
        self.cur_log_idx += 1

    def _build_click_tree(self, task_url):
        c_tree = defaultdict(list)
        c_logs = ExtensionLog.objects(user=self.user, task_url=task_url, action='CLICK')
        for l in c_logs:
            message = json.loads(l.message)
            if message.get('type') != 'anchor':
                continue
            c_tree[message.get('href')].append((l.site, l.timestamp))
        return c_tree

    def _build_redirect_tree(self, task_url):
        r_tree = defaultdict(list)
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
                        r_tree[l.site].append((last_end.site, last_end.timestamp))
                        last_end = None
        return r_tree

    @staticmethod
    def _is_serp(url):
        return url.startswith('http://10.129.248.120:8000/search_api/')

    @staticmethod
    def _get_query_from_site(site_url):
        site_url = site_url.encode('utf8')
        try:
            query_part = BehaviorLogExtractor._query_pattern.search(site_url).group(1)
            return urllib.unquote_plus(query_part).decode('utf8')
        except AttributeError:
            print 'No query in %s' % site_url


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
                    username=self.user.username,
                    task_url=task_url,

                    url=l.site,
                    query=BehaviorLogExtractor._get_query_from_site(l.site),
                    page_num=BehaviorLogExtractor._get_page_num_from_site(l.site),
                    start_time=l.timestamp,

                    html=message.get('html'),
                    mhtml=message.get('mhtml'),
                    visible_elements=json.dumps(message.get('visible_elements'), ensure_ascii=False),

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

    @staticmethod
    def clean_database():
        TaskSession.objects().delete()
        Query.objects().delete()
        LandingPage.objects().delete()
        SERPPage.objects().delete()
        ViewPort.objects().delete()
        Hover.objects().delete()
        Click.objects().delete()
        MouseMovement.objects().delete()
        Fixation.objects().delete()

    def save_task_session_to_database(self, task_url):
        task_session = self.extract_task_session(task_url)
        utils.save_doc_to_db(task_session)


def test():
    extractor = BehaviorLogExtractor('2015012338')
    extractor.clean_databse()
    for i in range(1, 7):
        extractor.save_task_session_to_database('/exp_domain_expertise/%d/' % i)


def insert_to_database():
    import os
    filenames = os.listdir('./eyetracking_log/')
    usernames = [f.split('.')[0] for f in filenames]
    BehaviorLogExtractor.clean_database()
    for username in usernames:
        extractor = BehaviorLogExtractor(username)
        for i in range(1, 7):
            extractor.save_task_session_to_database('/exp_domain_expertise/%d/' % i)


def iter_user_and_task():
    import os
    filenames = os.listdir('./eyetracking_log/')
    usernames = [f.split('.')[0] for f in filenames]
    for username in usernames:
        for i in range(1, 7):
            yield username, '/exp_domain_expertise/%d/' % i


def extract_viewport_info_str_from_page(eye_reader, page):
    ret = []
    ret.append('%s\t%s\t%s\t%d' % (page.username, page.task_url, page.url, len(page.viewports)))
    for vp in page.viewports:
        ret[-1] += '\t%d %d %d' % (vp.x_pos, vp.y_pos, eye_reader.get_recording_timestamp(vp.start_time))
    for p in page.clicked_pages:
        ret += extract_viewport_info_str_from_page(eye_reader, p)
    return ret


def extract_viewport_info_str():
    import os
    filenames = os.listdir('./eyetracking_log/')
    usernames = [f.split('.')[0] for f in filenames]
    BehaviorLogExtractor.clean_database()

    ret = []
    for username in usernames:
        extractor = BehaviorLogExtractor(username)
        for i in range(1, 7):
            task_session = extractor.extract_task_session('/exp_domain_expertise/%d/' % i)
            for q in task_session.queries:
                for p in q.pages:
                    ret += extract_viewport_info_str_from_page(extractor.eye_reader, p)
    with open('./tmp/viewports.txt', 'w') as fout:
        for idx, line in enumerate(ret):
            print >>fout, '%d\t%s' % (idx, line)






