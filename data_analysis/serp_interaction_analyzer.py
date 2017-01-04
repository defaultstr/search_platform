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
from search_api.bing_api.BingSE import parse_Bing_serp
from exp_domain_expertise.models import *
from fixation_time_analyzer import FixationTimeAnalyzerBase
from behavior_log.models import *
from collections import Counter, defaultdict
import numpy as np


class SERPInteractionAnalyzerBase(DataAnalyzer):
    def __init__(self, user_list=None):
        super(SERPInteractionAnalyzerBase, self).__init__(user_list=user_list)


class SERPInteractionAnalyzer(SERPInteractionAnalyzerBase):
    def __init__(self, user_list=None):
        super(SERPInteractionAnalyzer, self).__init__(user_list=user_list)

    @query_var_decorator
    def satisfaction(self, row, task_session, query):
        return query.satisfaction_score

    @query_var_decorator
    def num_pages(self, row, task_session, query):
        return len(query.pages)

    @query_var_decorator
    def num_clicks(self, row, task_session, query):
        ret = 0
        for p in query.pages:
            for vp in p.viewports:
                ret += len(vp.clicks)
        return ret

    @query_var_decorator
    def num_mouse_movements(self, row, task_session, query):
        ret = 0
        for p in query.pages:
            for vp in p.viewports:
                ret += len(vp.mouse_movements)
        return ret

    @query_var_decorator
    def num_hovers(self, row, task_session, query):
        ret = 0
        for p in query.pages:
            for vp in p.viewports:
                ret += len(vp.hovers)
        return ret

    @query_var_decorator
    def num_fixations(self, row, task_session, query):
        ret = 0
        for p in query.pages:
            for vp in p.viewports:
                ret += len(vp.fixations)
        return ret

    @query_var_decorator
    def dwell_time(self, row, task_session, query):
        return query.end_time - query.start_time

    @query_var_decorator
    def serp_time(self, row, task_session, query):
        ret = 0
        for p in query.pages:
            for vp in p.viewports:
                ret += vp.end_time - vp.start_time
        return ret

    @query_var_decorator
    def serp_time_per_page(self, row, task_session, query):
        return self.serp_time(row, task_session, query) / self.num_pages(row, task_session, query)

    @query_var_decorator
    def num_tab_switch(self, row, task_session, query):
        ret = 0
        for vp0, vp1 in self._get_all_viewport_transitions(task_session, query):
            if vp1.start_time - vp0.end_time > 100 and vp1.y_pos == vp0.y_pos:
                ret += 1
        return ret

    @query_var_decorator
    def num_scroll(self, row, task_session, query):
        ret = 0
        for vp0, vp1 in self._get_all_viewport_transitions(task_session, query):
            if vp0.y_pos != vp1.y_pos:
                ret += 1
        return ret

    @query_var_decorator
    def num_scroll_down(self, row, task_session, query):
        ret = 0
        for vp0, vp1 in self._get_all_viewport_transitions(task_session, query):
            if vp1.y_pos > vp0.y_pos:
                ret += 1
        return ret

    @query_var_decorator
    def num_scroll_up(self, row, task_session, query):
        ret = 0
        for vp0, vp1 in self._get_all_viewport_transitions(task_session, query):
            if vp1.y_pos < vp0.y_pos:
                ret += 1
        return ret

    @query_var_decorator
    def scroll_bottom_pos(self, row, task_session, query):
        ret = 0
        for vp0, vp1 in self._get_all_viewport_transitions(task_session, query):
            ret = vp1.y_pos if vp1.y_pos > ret else ret
        return ret

    @query_var_decorator
    def min_click_depth(self, row, task_session, query):
        depth = self._get_click_depth(query)
        if len(depth) == 0:
            return 1.0
        return np.min(depth)

    @query_var_decorator
    def avg_click_depth(self, row, task_session, query):
        depth = self._get_click_depth(query)
        if len(depth) == 0:
            return 1.0
        return np.average(depth)

    @query_var_decorator
    def max_click_depth(self, row, task_session, query):
        depth = self._get_click_depth(query)
        if len(depth) == 0:
            return 1.0
        return np.max(depth)

    @staticmethod
    def _get_click_depth(query):
        ret = []
        url_depth = SERPInteractionAnalyzer._get_url_depth(query)
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
            url_depth_in_page = SERPInteractionAnalyzer._get_url_depth_in_page(page)
            for url in url_depth_in_page:
                ret[url] = page_idx * 10 + url_depth_in_page[url]
        return ret

    @staticmethod
    def _get_url_depth_in_page(page):
        results = parse_Bing_serp(page.html)
        ret = {r['url']: idx+1 for idx, r in enumerate(results)}
        return ret

    def _get_all_viewport_transitions(self, task_session, query):
        for p in query.pages:
            for vp0, vp1 in zip(p.viewports[:-1], p.viewports[1:]):
                yield vp0, vp1


def get_serp_interaction_df():
    a = SERPInteractionAnalyzer()
    a.check_connection()
    df = a.get_query_data_df()
    df.to_pickle(ROOT + '/tmp/serp_interaction.dataframe')

    print DataAnalyzer.dataframe_stat(df)

if __name__ == '__main__':
    get_serp_interaction_df()
