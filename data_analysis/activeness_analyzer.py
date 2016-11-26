#!/user/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

if __name__ == '__main__':
    import sys
    ROOT = sys.argv[1]
    sys.path.insert(0, ROOT)
    from mongoengine import *
    connect('search_platform')

from data_analyzer import DataAnalyzer, variable_decorator
from exp_domain_expertise.models import *
from behavior_log.models import *


class ActivenessAnalyzer(DataAnalyzer):
    def __init__(self, user_list=None):
        super(ActivenessAnalyzer, self).__init__(user_list=user_list)

    @variable_decorator
    def total_task_time(self, row, task_session):
        assert isinstance(task_session, TaskSession)
        return task_session.end_time - task_session.start_time

    @variable_decorator
    def query_num(self, row, task_session):
        assert isinstance(task_session, TaskSession)
        return len(task_session.queries)

    @variable_decorator
    def serp_num(self, row, task_session):
        assert isinstance(task_session, TaskSession)
        ret = 0
        for q in task_session.queries:
            assert isinstance(q, Query)
            ret += len(q.pages)
        return ret

    @variable_decorator
    def fixation_num(self, row, task_session):
        assert isinstance(task_session, TaskSession)
        ret = 0
        for q in task_session.queries:
            assert isinstance(q, Query)
            for p in q.pages:
                ret += self._get_page_fixation_num(p)
        return ret

    @variable_decorator
    def click_num(self, row, task_session):
        assert isinstance(task_session, TaskSession)
        ret = 0
        for q in task_session.queries:
            assert isinstance(q, Query)
            for p in q.pages:
                ret += self._get_page_click_num(p)
        return ret

    @variable_decorator
    def serp_time_percentage(self, row, task_session):
        assert isinstance(task_session, TaskSession)
        serp_time = 0.0
        for q in task_session.queries:
            assert isinstance(q, Query)
            for p in q.pages:
                assert isinstance(p, SERPPage)
                serp_time += p.end_time - p.start_time
        return serp_time / (task_session.end_time - task_session.start_time)

    @variable_decorator
    def serp_num_per_query(self, row, task_session):
        return 1.0 * self.serp_num(row, task_session) / self.query_num(row, task_session)

    @variable_decorator
    def fixation_num_per_query(self, row, task_session):
        return 1.0 * self.fixation_num(row, task_session) / self.query_num(row, task_session)

    @variable_decorator
    def click_num_per_query(self, row, task_session):
        return 1.0 * self.click_num(row, task_session) / self.query_num(row, task_session)

    @variable_decorator
    def time_per_query(self, row, task_session):
        return 1.0 * self.total_task_time(row, task_session) / self.query_num(row, task_session)

    def _get_page_fixation_num(self, page):
        assert isinstance(page, SERPPage) or isinstance(page, LandingPage)
        ret = 0
        for v in page.viewports:
            assert isinstance(v, ViewPort)
            ret += len(v.fixations)
        for p in page.clicked_pages:
            ret += self._get_page_fixation_num(p)
        return ret

    def _get_page_click_num(self, page):
        assert isinstance(page, SERPPage) or isinstance(page, LandingPage)
        ret = 0
        for v in page.viewports:
            assert isinstance(v, ViewPort)
            ret += len(v.clicks)
        for p in page.clicked_pages:
            ret += self._get_page_click_num(p)
        return ret


if __name__ == '__main__':
    a = ActivenessAnalyzer()
    a.check_connection()
    df = a.get_data_df()
    df.to_pickle(ROOT + '/tmp/activeness.dataframe')

    print DataAnalyzer.dataframe_stat(df)


