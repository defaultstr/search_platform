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
from fixation_time_analyzer import FixationTimeAnalyzerBase
from behavior_log.models import *
from collections import Counter, defaultdict
import numpy as np


class QueryAnalyzerBase(DataAnalyzer):
    def __init__(self, user_list=None):
        super(QueryAnalyzerBase, self).__init__(user_list=user_list)

    def _get_term_lists(self, queries):
        return [self._word_segment(q.query) for q in queries]


class QueryAnalyzer(QueryAnalyzerBase):
    def __init__(self, user_list=None):
        super(QueryAnalyzer, self).__init__(user_list=user_list)
        self.df = Counter()
        self.doc_num = 0
        self.idf = None

        self.init_idf()

    def init_idf(self):
        for idx, ts in enumerate(self.task_sessions):
            #assert isinstance(ts, TaskSession)
            task = self._get_task_from_task_session(ts)
            terms = self._word_segment(task.description)
            self.df.update(terms)
            self.doc_num += 1
            print 'Processing Task Session #%d' % idx

            for page in self._get_all_pages_in_task_session(ts):
                terms = set(self._word_segment(
                    self._get_text_from_page(page)
                ))
                self.df.update(terms)
                self.doc_num += 1

            for q in ts.queries:
                #assert isinstance(q, Query)
                terms = set(self._word_segment(q.query))
                self.df.update(terms)
                self.doc_num += 1

        self.idf = defaultdict(lambda: np.log(self.doc_num))
        self.idf.update({term: np.log(1.0*self.doc_num/self.df[term]) for term in self.df})
        print 'Vocabulary Size: %d' % len(self.idf)

    @query_var_decorator
    def query_character_length(self, row, task_session, query):
        #assert isinstance(query, Query)
        return len(query.query)

    @query_var_decorator
    def query_word_length(self, row, task_session, query):
        #assert isinstance(query, Query)
        return len(self._word_segment(query.query))

    @query_var_decorator
    def avg_idf(self, row, task_session, query):
        #assert isinstance(query, Query)
        terms = self._word_segment(query.query)
        return np.average([self.idf[term] for term in terms])


    @query_var_decorator
    def max_idf(self, row, task_session, query):
        #assert isinstance(query, Query)
        terms = self._word_segment(query.query)
        return np.max([self.idf[term] for term in terms])

    @query_var_decorator
    def from_task_desc(self, row, task_session, query):
        task = self._get_task_from_task_session(task_session)
        task_desc_terms = set(self._word_segment(task.description))
        q_terms = self._word_segment(query.query)
        return 1.0 * np.sum([1.0 for t in q_terms if t in task_desc_terms]) / len(q_terms)


class QueryReformAnalyzer(QueryAnalyzerBase):
    def __init__(self, user_list=None):
        super(QueryReformAnalyzer, self).__init__(user_list=user_list)

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

    #@query_var_decorator
    def repeat(self, row, task_session, query):
        if row.query_idx == 0:
            return 0
        term_lists = self._get_term_lists(task_session.queries)
        for i in range(0, row.query_idx):
            if set(term_lists[i]) == set(term_lists[row.query_idx]):
                return 1
        return 0

    @query_var_decorator
    def jaccard_dist(self, row, task_session, query):
        if row.query_idx == 0:
            return 0.0
        term_lists = self._get_term_lists(task_session.queries)
        q0 = set(term_lists[row.query_idx-1])
        q1 = set(term_lists[row.query_idx])
        return 1.0 * len(q0.intersection(q1)) / len(q0.union(q1))

    @query_var_decorator
    def edit_dist(self, row, task_session, query):
        if row.query_idx == 0:
            return 0.0
        term_lists = self._get_term_lists(task_session.queries)
        q0 = term_lists[row.query_idx-1]
        q1 = term_lists[row.query_idx]
        return self._edit_dist(q0, q1)

    #@query_var_decorator
    def query_term_from_serp(self, row, task_session, query):
        return self._query_term_from_source(row, task_session,
                                          start_time=None,
                                          end_time=query.start_time,
                                          page_type=SERPPage)

    #@query_var_decorator
    def query_term_from_landing_page(self, row, task_session, query):
        return self._query_term_from_source(row, task_session,
                                          start_time=None,
                                          end_time=query.start_time,
                                          page_type=LandingPage)

    #@query_var_decorator
    def query_term_from_last_serp(self, row, task_session, query):
        if row.query_idx == 0:
            return 0.0

        return self._query_term_from_source(row, task_session,
                                          start_time=task_session.queries[row.query_idx-1].start_time,
                                          end_time=query.start_time,
                                          page_type=SERPPage)


    #@query_var_decorator
    def query_term_from_last_landing_page(self, row, task_session, query):
        if row.query_idx == 0:
            return 0.0

        return self._query_term_from_source(row, task_session,
                                          start_time=task_session.queries[row.query_idx-1].start_time,
                                          end_time=query.start_time,
                                          page_type=LandingPage)

    #@query_var_decorator
    def query_term_from_prior_knowledge(self, row, task_session, query):
        if row.query_idx == 0:
            return 0.0
        term_lists = self._get_term_lists(task_session.queries)
        q_terms = term_lists[row.query_idx]

        task = self._get_task_from_task_session(task_session)
        desc_terms = set(self._word_segment(task.description))
        f_terms = self._get_term_from_source(task_session,
                                             start_time=None,
                                             end_time=query.start_time)
        s_terms = desc_terms.union(f_terms)
        return sum(1.0 for t in q_terms if t not in s_terms) / len(q_terms)

    @staticmethod
    def _get_term_from_source(task_session,
                              threshold=100,
                              start_time=None,
                              end_time=None,
                              page_type=None):

        f_time_on_terms = FixationTimeAnalyzerBase._get_fixation_time(
            task_session,
            start_time=start_time,
            end_time=end_time,
            page_type=page_type,
        )
        s_terms = set(t for t in f_time_on_terms if f_time_on_terms[t] > threshold)
        return s_terms

    def _query_term_from_source(self, row, task_session,
                              start_time=None,
                              end_time=None,
                              page_type=None):
        if row.query_idx == 0:
            return 0.0
        term_lists = self._get_term_lists(task_session.queries)
        q_terms = term_lists[row.query_idx]
        s_terms = self._get_term_from_source(task_session,
                                             start_time=start_time,
                                             end_time=end_time,
                                             page_type=page_type)
        return sum(1.0 for t in q_terms if t in s_terms) / len(q_terms)

    @staticmethod
    def _edit_dist(s0, s1):
        dist = []
        n = len(s0)
        m = len(s1)
        for i in range(n):
            dist.append([0] * m)

        if n == 0:
            return m
        if m == 0:
            return n

        if s0[0] != s1[0]:
            dist[0][0] = 1

        for k in range(1, n + m - 1):
            start = max(0, k-m+1)
            end = min(k+1, n)
            for i in range(start, end):
                j = k - i
                x = max(m, n)
                if i > 0:
                    x = min(dist[i-1][j] + 1, x)
                if j > 0:
                    x = min(dist[i][j-1] + 1, x)
                if i > 0 and j > 0:
                    if s0[i] == s1[j]:
                        x = min(dist[i-1][j-1], x)
                    else:
                        x = min(dist[i-1][j-1] + 1, x)
                dist[i][j] = x
        return dist[n-1][m-1]

    def compute_other_session_features(self):
        self.session_df['num_reform'] = [
            len(
                self.query_df[(self.query_df.uid == row.uid) & (self.query_df.task_id == row.task_id)]
            ) - 1.0
            for idx, row in self.session_df.iterrows()
        ]

        self.session_df['num_add'] = [
            np.sum(
                self.query_df[(self.query_df.uid == row.uid) & (self.query_df.task_id == row.task_id)]['has_added_term']
            )
            for idx, row in self.session_df.iterrows()
        ]
        self.session_df['add_rate'] = 1.0 * self.session_df.num_add / self.session_df.num_reform

        self.session_df['num_remove'] = [
            np.sum(
                self.query_df[(self.query_df.uid == row.uid) & (self.query_df.task_id == row.task_id)]['has_removed_term']
            )
            for idx, row in self.session_df.iterrows()
        ]
        self.session_df['remove_rate'] = 1.0 * self.session_df.num_remove / self.session_df.num_reform

        self.session_df['num_retain'] = [
            np.sum(
                self.query_df[(self.query_df.uid == row.uid) & (self.query_df.task_id == row.task_id)]['has_retained_term']
            )
            for idx, row in self.session_df.iterrows()
        ]
        self.session_df['retain_rate'] = 1.0 * self.session_df.num_retain / self.session_df.num_reform

        self.session_df['num_generalization'] = [
            np.sum(
                self.query_df[(self.query_df.uid == row.uid) & (self.query_df.task_id == row.task_id)]['generalization']
            )
            for idx, row in self.session_df.iterrows()
        ]
        self.session_df['generalization_rate'] = 1.0 * self.session_df.num_generalization / self.session_df.num_reform

        self.session_df['num_specialization'] = [
            np.sum(
                self.query_df[(self.query_df.uid == row.uid) & (self.query_df.task_id == row.task_id)]['specialization']
            )
            for idx, row in self.session_df.iterrows()
        ]
        self.session_df['specialization_rate'] = 1.0 * self.session_df.num_specialization / self.session_df.num_reform

        self.session_df['num_substitution'] = [
            np.sum(
                self.query_df[(self.query_df.uid == row.uid) & (self.query_df.task_id == row.task_id)]['substitution']
            )
            for idx, row in self.session_df.iterrows()
        ]
        self.session_df['substitution_rate'] = 1.0 * self.session_df.num_substitution / self.session_df.num_reform

        self.session_df = self.session_df.fillna(0.0)


def get_query_df():
    a = QueryAnalyzer()
    a.check_connection()
    df = a.get_query_data_df()
    df.to_pickle(ROOT + '/tmp/query.dataframe')

    print DataAnalyzer.dataframe_stat(df)


def get_query_reform_df():
    a = QueryReformAnalyzer()
    a.check_connection()
    df = a.get_query_data_df()
    df = a.get_data_df()
    df.to_pickle(ROOT + '/tmp/query_reform_session.dataframe')

    print DataAnalyzer.dataframe_stat(df)

if __name__ == '__main__':
    #get_query_df()
    get_query_reform_df()
