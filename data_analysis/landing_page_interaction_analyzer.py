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


class LandingPageInteractionAnalyzerBase(DataAnalyzer):
    def __init__(self, user_list=None):
        super(LandingPageInteractionAnalyzerBase, self).__init__(user_list=user_list)

    def _get_usefulness_scores(self, query):
        _, all_pages = self._get_all_landing_pages(query)
        return [p.usefulness_score for p in all_pages]

    @staticmethod
    def _get_all_landing_pages_from_task_session(task_session):
        ret = []
        for q in task_session.queries:
            ret += LandingPageInteractionAnalyzerBase._get_all_landing_pages(q)[1]
        return ret

    @staticmethod
    def _get_all_landing_pages(query):
        def _get_clicked_pages(page):
            ret = []
            for p in page.clicked_pages:
                ret.append(p)
                ret += _get_clicked_pages(p)
            return ret

        first_level_landing_pages = []
        all_landing_pages = []
        for p in query.pages:
            for cp in p.clicked_pages:
                first_level_landing_pages.append(cp)
                all_landing_pages.append(cp)
                all_landing_pages += _get_clicked_pages(cp)

        return first_level_landing_pages, all_landing_pages


class LandingPageAnalyzer(LandingPageInteractionAnalyzerBase):
    def __init__(self, user_list=None):
        super(LandingPageAnalyzer, self).__init__(user_list=user_list)

    @variable_decorator
    def num_landing_pages(self, row, task_session):
        return len(self._get_all_landing_pages_from_task_session(task_session))

    def _num_landing_pages_of_score(self, row, task_session, score=1):
        return len([
                       p
                       for p in self._get_all_landing_pages_from_task_session(task_session)
                       if p.usefulness_score == score
        ])

    @variable_decorator
    def score_4_ratio(self, row, task_session):
        return 1.0 * self._num_landing_pages_of_score(row, task_session, 4) / self.num_landing_pages(row, task_session)

    @variable_decorator
    def score_3_ratio(self, row, task_session):
        return 1.0 * self._num_landing_pages_of_score(row, task_session, 3) / self.num_landing_pages(row, task_session)

    @variable_decorator
    def score_2_ratio(self, row, task_session):
        return 1.0 * self._num_landing_pages_of_score(row, task_session, 2) / self.num_landing_pages(row, task_session)

    @variable_decorator
    def score_1_ratio(self, row, task_session):
        return 1.0 * self._num_landing_pages_of_score(row, task_session, 1) / self.num_landing_pages(row, task_session)

    def _get_dwell_times_on_landing_pages(self, task_session):
        ret = []
        for p in self._get_all_landing_pages_from_task_session(task_session):
            ret.append(np.sum((vp.end_time-vp.start_time) for vp in p.viewports))
        return ret

    @variable_decorator
    def sat_click_ratio(self, row, task_session):
        dwell_times = self._get_dwell_times_on_landing_pages(task_session)
        return 1.0 * len([dt for dt in dwell_times if dt >= 30*1000]) / len(dwell_times)

    @variable_decorator
    def short_click_ratio(self, row, task_session):
        dwell_times = self._get_dwell_times_on_landing_pages(task_session)
        return 1.0 * len([dt for dt in dwell_times if dt < 5*1000]) / len(dwell_times)

    def _get_domains(self, task_session):
        ret = []
        for p in self._get_all_landing_pages_from_task_session(task_session):
            assert isinstance(p, LandingPage)
            url = p.url
            # remove http://
            url = url[url.find('://')+3:]
            # extract domain
            domain = url.split('/')[0]
            # extract first level domain
            if len(domain.split('.')) > 2:
                domain = '.'.join(domain.split('.')[-2:])
            ret.append(domain)
        return ret

    @variable_decorator
    def num_unique_domain(self, row, task_session):
        return len(set(self._get_domains(task_session)))

    @variable_decorator
    def domain_richness(self, row, task_session):
        return 1.0 * self.num_unique_domain(row, task_session) \
               / len(self._get_all_landing_pages_from_task_session(task_session))


class LandingPageInteractionAnalyzer(LandingPageInteractionAnalyzerBase):
    def __init__(self, user_list=None):
        super(LandingPageInteractionAnalyzer, self).__init__(user_list=user_list)

    @query_var_decorator
    def time_on_landing_page(self, row, task_session, query):
        ret = 0.0
        for p in self._get_all_landing_pages(query)[1]:
            for vp in p.viewports:
                if vp.start_time > query.end_time:
                    continue
                ret += vp.end_time - vp.start_time
        return ret

    @query_var_decorator
    def time_per_landing_page(self, row, task_session, query):
        time_on_page = defaultdict(lambda: 0.0)
        for p in self._get_all_landing_pages(query)[1]:
            for vp in p.viewports:
                if vp.start_time > query.end_time:
                    continue
                time_on_page[p.url] += vp.end_time - vp.start_time
        if len(time_on_page) == 0:
            return 0.0

        return 1.0 * sum(time_on_page.values()) / len(time_on_page)

    @query_var_decorator
    def num_fixations(self, row, task_session, query):
        ret = 0.0
        for p in self._get_all_landing_pages(query)[1]:
            for vp in p.viewports:
                if vp.start_time > query.end_time:
                    continue
                ret += len(vp.fixations)
        return ret

    @query_var_decorator
    def num_mouse_movements(self, row, task_session, query):
        ret = 0.0
        for p in self._get_all_landing_pages(query)[1]:
            for vp in p.viewports:
                if vp.start_time > query.end_time:
                    continue
                ret += len(vp.mouse_movements)
        return ret


    @query_var_decorator
    def min_usefulness(self, row, task_session, query):
        scores = self._get_usefulness_scores(query)
        if len(scores) == 0:
            return 0
        return np.min(scores)

    @query_var_decorator
    def max_usefulness(self, row, task_session, query):
        scores = self._get_usefulness_scores(query)
        if len(scores) == 0:
            return 0
        return np.max(scores)

    @query_var_decorator
    def avg_usefulness(self, row, task_session, query):
        scores = self._get_usefulness_scores(query)
        if len(scores) == 0:
            return 0
        return np.average(scores)

    @query_var_decorator
    def num_tab_switch(self, row, task_session, query):
        all_viewports = self._get_all_viewports(task_session, query)
        ret = 0
        for pvp0, pvp1 in zip(all_viewports[:-1], all_viewports[1:]):
            p0, vp0 = pvp0
            p1, vp1 = pvp1
            if p0 != p1 and isinstance(p0, LandingPage):
                ret += 1
        return ret

    @query_var_decorator
    def num_scroll(self, row, task_session, query):
        all_viewports = self._get_all_viewports(task_session, query)
        ret = 0
        for pvp0, pvp1 in zip(all_viewports[:-1], all_viewports[1:]):
            p0, vp0 = pvp0
            p1, vp1 = pvp1
            if p0 == p1 and isinstance(p0, LandingPage) and vp1.start_time - vp0.end_time < 100:
                ret += 1
        return ret

    @query_var_decorator
    def num_click_pages(self, row, task_session, query):
        first_level_pages, all_pages = self._get_all_landing_pages(query)
        return len(all_pages)

    @query_var_decorator
    def avg_branchiness(self, row, task_session, query):
        first_level_pages, all_pages = self._get_all_landing_pages(query)
        x = sum([len(p.clicked_pages) for p in all_pages])
        if x == 0:
            return 0
        else:
            return 1.0 * x / len([p for p in all_pages if len(p.clicked_pages) > 0])

    @query_var_decorator
    def max_branchiness(self, row, task_session, query):
        first_level_pages, all_pages = self._get_all_landing_pages(query)
        if len(all_pages) == 0:
            return 0
        return max([len(p.clicked_pages) for p in all_pages])

    @query_var_decorator
    def depth_of_click_tree(self, row, task_session, query):
        def _get_depth(page):
            max_depth = 0
            for p in page.clicked_pages:
                max_depth = max(max_depth, _get_depth(p))
            return max_depth + 1

        max_depth = 0
        for p in query.pages:
            for cp in p.clicked_pages:
                max_depth = max(max_depth, _get_depth(cp))
        return max_depth


    def _get_all_viewports(self, task_session, query):
        viewports = []
        for p in query.pages:
            for vp in p.viewports:
                viewports.append((p, vp))
        for p in self._get_all_landing_pages(query)[1]:
            for vp in p.viewports:
                viewports.append((p, vp))
        return sorted(viewports, key=lambda x: x[1].start_time)


def group_by_landing_page_url():
    a = LandingPageInteractionAnalyzer()
    a.check_connection()
    ret = defaultdict(list)
    def visit_function(row, task_session, query):
        landing_pages = a._get_all_landing_pages(query)[1]
        for p in landing_pages:
            dwell_time = p.end_time - p.start_time

            viewports = p.viewports
            num_viewports = len(viewports)
            page_time = sum([vp.end_time-vp.start_time for vp in viewports])
            num_fixations = sum([len(vp.fixations) for vp in viewports])
            max_scroll_y = max([0]+[vp.y_pos for vp in viewports])

            ret[p.url].append((row.in_domain, p.usefulness_score,
                               dwell_time, num_viewports, page_time, num_fixations, max_scroll_y))
        return 0
    a.add_query_variable(variable_name=None, extract_function=visit_function)
    return ret


def compare_expert_and_non_expert(landing_page_info):
    expert_info = {}
    non_expert_info = {}
    for url in landing_page_info:
        info_list = landing_page_info[url]
        if 1 not in [info[0] for info in info_list] or 0 not in [info[0] for info in info_list]:
            continue
        expert_info[url] = [np.average([info[i] for info in info_list if info[0] == 1]) for i in range(1, 7)]
        non_expert_info[url] = [np.average([info[i] for info in info_list if info[0] == 0]) for i in range(1, 7)]
    return expert_info, non_expert_info


def get_landing_page_interaction_df():
    a = LandingPageInteractionAnalyzer()
    a.check_connection()
    df = a.get_query_data_df()
    df.to_pickle(ROOT + '/tmp/landing_page_interaction.dataframe')

    print DataAnalyzer.dataframe_stat(df)


if __name__ == '__main__':
    a = LandingPageAnalyzer()
    a.check_connection()
    df = a.get_data_df()
    df.to_pickle(ROOT + '/tmp/landing_page_session.dataframe')

    print DataAnalyzer.dataframe_stat(df)
