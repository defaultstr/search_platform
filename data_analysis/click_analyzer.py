#!/user/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

if __name__ == '__main__':
    import sys
    ROOT = sys.argv[1]
    sys.path.insert(0, ROOT)
    from mongoengine import *
    connect('search_platform')

from data_analyzer import DataAnalyzer, variable_decorator, query_var_decorator
from fixation_time_analyzer import FixationTimeAnalyzer
from search_api.bing_api.BingSE import parse_Bing_serp
from exp_domain_expertise.models import *
from fixation_time_analyzer import FixationTimeAnalyzerBase
from behavior_log.models import *
from collections import Counter, defaultdict
from BeautifulSoup import BeautifulSoup
import json
import numpy as np


class ClickAnalyzerBase(DataAnalyzer):
    def __init__(self, user_list=None):
        super(ClickAnalyzerBase, self).__init__(user_list=user_list)
        self.rank_click_and_examine_info = dict()

    @staticmethod
    def _get_click_depth(query):
        ret = []
        url_depth = ClickAnalyzer._get_url_depth(query)
        for p in query.pages:
            for vp in p.viewports:
                for click in vp.clicks:
                    if click.url in url_depth:
                        ret.append(url_depth[click.url])
        return ret

    @staticmethod
    def _get_url_depth(query):
        ret = {}
        for page_idx, page in enumerate(query.pages):
            url_depth_in_page = ClickAnalyzer._get_url_depth_in_page(page)
            for url in url_depth_in_page:
                ret[url] = page_idx * 10 + url_depth_in_page[url]
        return ret

    @staticmethod
    def _get_url_depth_in_page(page):
        results = parse_Bing_serp(page.html)
        ret = {r['url']: idx+1 for idx, r in enumerate(results)}
        return ret

    @staticmethod
    def startswith(long_str, short_str):
        if len(long_str) < len(short_str):
            return False
        if long_str.startswith(short_str):
            return True
        err_num = 0
        for idx, c in enumerate(short_str):
            if c != long_str[idx]:
                err_num += 1

        if 1.0 * err_num / len(short_str) < 0.1:
            return True
        else:
            return False

    @staticmethod
    def parse_serp(page):
        assert isinstance(page, SERPPage)
        results = parse_Bing_serp(page.html)
        ret = []
        elements = json.loads(page.visible_elements)
        for e in elements:
            e['text'] = e['text'].replace(' ', '')

        sorted_elements = sorted(elements, key=lambda e: -FixationTimeAnalyzer._get_area(e))
        """
        for e in sorted_elements:
            if e['left'] == 100:
                print e['text']
        """
        for idx, r in enumerate(results):
            rank = idx + 1
            url = r['url']
            title = BeautifulSoup(r['title'], convertEntities=BeautifulSoup.HTML_ENTITIES).text
            title = title.replace(' ', '')

            #print title
            snippet = BeautifulSoup(r['snippet']).text
            b_rect = [
                (e['left'], e['top'], e['right'], e['bottom'])
                for e in sorted_elements
                if ClickAnalyzerBase.startswith(e['text'], title) and (idx > 0 or e['bottom'] - e['top'] < 500)
                ][0]
            ret.append((rank, url, b_rect, title, snippet))
        return ret

    @staticmethod
    def get_examined_results(page, results, threshold=200):
        cum_time = np.zeros(len(results), dtype=int)
        for vp in page.viewports:
            for f in vp.fixations:
                for idx, r in enumerate(results):
                    left, top, right, bottom = r[2]
                    if left <= f.x_on_page <= right and top <= f.y_on_page <= right:
                        cum_time[idx] += f.duration
        return zip(
            [idx + (page.page_num-1) * 10 for idx in range(len(results))],
            [1 if x > threshold else 0 for x in cum_time],
            list(cum_time)
        )

    @staticmethod
    def get_clicked_results(page, results):
        clicked_urls = set()
        for vp in page.viewports:
            for c in vp.clicks:
                clicked_urls.add(c.url)
        return zip(
            [idx + (page.page_num-1) * 10 for idx in range(len(results))],
            [1 if r[1] in clicked_urls else 0 for r in results]
        )

    @staticmethod
    def get_dwell_time(page):
        return np.sum([vp.end_time - vp.start_time for vp in page.viewports])

    @staticmethod
    def get_usefulness_and_dwell_time(page, results):
        usefulness_score = defaultdict(lambda: 0)
        dwell_time = defaultdict(lambda: 0)
        #assert isinstance(page, SERPPage)
        for cp in page.clicked_pages:
            #assert isinstance(cp, LandingPage)
            usefulness_score[cp.url] = cp.usefulness_score
            dwell_time[cp.url] = ClickAnalyzerBase.get_dwell_time(cp)

        return zip(
            [idx + (page.page_num-1) * 10 for idx in range(len(results))],
            [usefulness_score[r[1]] for r in results],
            [dwell_time[r[1]] for r in results]
        )


    @staticmethod
    def get_rank_click_and_examine_info(query):
        ret = [] # [examine, examine_time, click, usefulness, dwell_time]
        ret = defaultdict(lambda: [0, 0, 0, 0, 0])
        for p in query.pages:
            results = ClickAnalyzerBase.parse_serp(p)
            for rank, e, f_time in ClickAnalyzerBase.get_examined_results(p, results):
                ret[rank][0], ret[rank][1] = e, f_time
            for rank, c in ClickAnalyzerBase.get_clicked_results(p, results):
                ret[rank][2] = c
            for rank, usefulness, dtime in ClickAnalyzerBase.get_usefulness_and_dwell_time(p, results):
                ret[rank][3] = usefulness
                ret[rank][4] = dtime
        return ret

    def _get_rank_click_and_examine(self, query):
        if query.id not in self.rank_click_and_examine_info:
            self.rank_click_and_examine_info[query.id] = self.get_rank_click_and_examine_info(query)
        return self.rank_click_and_examine_info[query.id]


class QueryEffectivenessAnalyzer(ClickAnalyzerBase):

    def __init__(self, user_list=None):
        super(QueryEffectivenessAnalyzer, self).__init__(user_list=user_list)


class ClickAnalyzer(ClickAnalyzerBase):
    def __init__(self, user_list=None):
        super(ClickAnalyzer, self).__init__(user_list=user_list)

    #Examine
    @variable_decorator
    def num_examined_results_per_query(self, row, task_session):
        return np.nanmean([
            np.sum([val[0] for val in self._get_rank_click_and_examine(q).itervalues()])
            for q in task_session.queries
        ])

    @variable_decorator
    def avg_examined_rank(self, row, task_session):
        return np.nanmean([
            np.mean([
                rank for rank, val in self._get_rank_click_and_examine(q).iteritems() if val[0] == 1
            ])
            for q in task_session.queries
        ])


    @variable_decorator
    def max_examined_rank(self, row, task_session):
        return np.nanmean([
            np.max([
                rank for rank, val in self._get_rank_click_and_examine(q).iteritems() if val[0] == 1
            ] + [0])
            for q in task_session.queries
        ])

    #Click
    @variable_decorator
    def num_clicks_per_query(self, row, task_session):
        return np.nanmean([
            np.sum([val[2] for val in self._get_rank_click_and_examine(q).itervalues()])
            for q in task_session.queries
        ])

    @variable_decorator
    def avg_clicked_rank(self, row, task_session):
        return np.nanmean([
            np.mean([
                rank for rank, val in self._get_rank_click_and_examine(q).iteritems() if val[2] == 1
            ])
            for q in task_session.queries
        ])

    @variable_decorator
    def max_clicked_rank(self, row, task_session):
        return np.nanmean([
            np.max([
                rank for rank, val in self._get_rank_click_and_examine(q).iteritems() if val[2] == 1
            ] + [0])
            for q in task_session.queries
        ])

def get_click_df():
    a = ClickAnalyzer()
    a.check_connection()
    df = a.get_data_df()
    df.to_pickle(ROOT + '/tmp/click.dataframe')

    print DataAnalyzer.dataframe_stat(df)

if __name__ == '__main__':
    get_click_df()
