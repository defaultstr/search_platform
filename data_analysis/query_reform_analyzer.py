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
from exp_domain_expertise.models import *
from behavior_log.models import *


class QueryReformAnalyzer(DataAnalyzer):
    def __init__(self, user_list=None):
        super(QueryReformAnalyzer, self).__init__(user_list=user_list)

    @query_var_decorator
    def query_character_length(self, row, task_session, query):
        assert isinstance(query, Query)
        return len(query.query)

    @query_var_decorator
    def query_word_length(self, row, task_session, query):
        assert isinstance(query, Query)
        return len(self._word_segment(query.query))

    @query_var_decorator
    def has_added_term(self, row, task_session, query):
        if row.query_idx == 0:
            return 0
        term_lists = self._get_term_lists(task_session.queries)
        if len(set(term_lists[row.query_idx]) - set(term_lists[row.query_idx - 1])) > 0:
            return 1
        else:
            return 0

    @query_var_decorator
    def has_removed_term(self, row, task_session, query):
        if row.query_idx == 0:
            return 0
        term_lists = self._get_term_lists(task_session.queries)
        if len(set(term_lists[row.query_idx - 1]) - set(term_lists[row.query_idx])) > 0:
            return 1
        else:
            return 0

    @query_var_decorator
    def has_retained_term(self, row, task_session, query):
        if row.query_idx == 0:
            return 0
        term_lists = self._get_term_lists(task_session.queries)
        if len(set(term_lists[row.query_idx - 1]).intersection(set(term_lists[row.query_idx]))) > 0:
            return 1
        else:
            return 0

    @query_var_decorator
    def generalization(self, row, task_session, query):
        return self.has_retained_term(row, task_session, query) * \
                self.has_removed_term(row, task_session, query) * \
                (1 - self.has_added_term(row, task_session, query))

    @query_var_decorator
    def specialization(self, row, task_session, query):
        return self.has_retained_term(row, task_session, query) * \
                (1 - self.has_removed_term(row, task_session, query)) * \
                self.has_added_term(row, task_session, query)

    @query_var_decorator
    def substitution(self, row, task_session, query):
        return self.has_retained_term(row, task_session, query) * \
                self.has_removed_term(row, task_session, query) * \
                self.has_added_term(row, task_session, query)

    @query_var_decorator
    def repeat(self, row, task_session, query):
        if row.query_idx == 0:
            return 0
        term_lists = self._get_term_lists(task_session.queries)
        for i in range(0, row.query_idx):
            if set(term_lists[i]) == set(term_lists[row.query_idx]):
                return 1
        return 0

    def _get_term_lists(self, queries):
        return [self._word_segment(q.query) for q in queries]


if __name__ == '__main__':
    a = QueryReformAnalyzer()
    a.check_connection()
    df = a.get_query_data_df()
    df.to_pickle(ROOT + '/tmp/query.dataframe')

    print DataAnalyzer.dataframe_stat(df)


