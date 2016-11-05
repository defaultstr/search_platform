#!/user/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

if __name__ == '__main__':
    import sys
    ROOT = sys.argv[1]
    sys.path.insert(0, ROOT)
    from mongoengine import *
    connect('search_platform')

from data_analyzer import DataAnalyzer
from mongoengine import connect
from behavior_log.models import *
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
import jieba
import json
import numpy as np
import pandas as pd
from exp_domain_expertise.utils import get_task_by_id


class FixationTimeAnalyzerBase(DataAnalyzer):

    def __init__(self, user_list=None):
        super(FixationTimeAnalyzerBase, self).__init__(user_list=user_list)

    def _get_query_and_add_terms(self, task_session, return_all=False):
        ret = []
        used_terms = set()
        for q in task_session.queries:
            query_terms = self._word_segment(q.query)
            add_terms = list(set(query_terms) - used_terms)
            if return_all:
                used_terms.update(add_terms)
                ret.append((q, add_terms))
            elif len(add_terms) > 0:
                used_terms.update(add_terms)
                ret.append((q, add_terms))
        return ret

    def _get_query_and_terms(self, task_session):
        return [(q, list(set(self._word_segment(q.query)))) for q in task_session.queries]

    def _get_query_and_new_terms(self, task_session, return_all=False):
        ret = []
        task_id = task_session.task_url.split('/')[2]
        task = get_task_by_id(task_id)
        task_desc_terms = set(self._word_segment(task.description))

        for q in task_session.queries:
            query_terms = self._word_segment(q.query)
            new_terms = list(set(query_terms) - task_desc_terms)
            if return_all:
                ret.append((q, new_terms))
            elif len(new_terms) > 0:
                ret.append((q, new_terms))
        return ret


    @staticmethod
    def _get_text_from_page(page):
        soup = BeautifulSoup(page.html)
        [s.extract() for s in soup('script')]
        [s.extract() for s in soup('style')]
        [s.extract() for s in soup('img')]
        return soup.text

    @staticmethod
    def _word_segment(text):
        """

        :param text: a text to segment
        :return: a list of segmented terms
        """
        return [w for w in jieba.cut(text) if w.strip() != '']

    @staticmethod
    def _get_term_freq(documents, terms=None):
        ret = Counter()
        for doc in documents:
            ret.update(FixationTimeAnalyzer._word_segment(doc))
        if terms:
            return {t: ret[t] for t in ret if t in terms}
        else:
            return dict(ret)

    @staticmethod
    def _fixation_on_elements(f, e, weight):
        if e['left'] <= f.x_on_page <= e['right'] and e['top'] <= f.y_on_page <= e['bottom']:
            if len(e['text']) > 100:
                return weight
            elif len(e['text']) > 20:
                return min(1000.0, weight)
            elif len(e['text']) > 5:
                return min(500.0, weight)
            else:
                return min(200.0, weight)
        else:
            return 0

    @staticmethod
    def _get_area(e):
        return (e['right'] - e['left']) * (e['bottom'] - e['top'])

    @staticmethod
    def _get_viewport_text(page, viewport):
        visible_elements = json.loads(page.visible_elements)
        sorted_elements = sorted(visible_elements, key=lambda e: FixationTimeAnalyzer._get_area(e))
        fixation_time = np.zeros((len(visible_elements)))
        for f in viewport.fixations:
            weight = f.duration
            for idx, e in enumerate(sorted_elements):
                f_time = FixationTimeAnalyzer._fixation_on_elements(f, e, weight)
                fixation_time[idx] += f_time
                weight -= f_time
                if weight <= 0:
                    break

        fixation_on_terms = defaultdict(lambda : 0.0)
        for idx, e in enumerate(sorted_elements):
            if fixation_time[idx] == 0.0:
                continue
            terms = FixationTimeAnalyzer._word_segment(e['text'])
            for t in terms:
                fixation_on_terms[t] += fixation_time[idx] * 1.0 / len(terms)
        return fixation_on_terms

    @staticmethod
    def _get_fixation_time(
            task_session,
            terms=None,
            start_time=None,
            end_time=None,
    ):
        ret = defaultdict(lambda : 0.0)
        pages = DataAnalyzer._get_all_pages_in_task_session(task_session)
        for p in pages:
            for vp in p.viewports:
                if start_time and vp.end_time <= start_time:
                    continue
                if end_time and vp.start_time >= end_time:
                    continue

                fixation_on_terms = FixationTimeAnalyzer._get_viewport_text(p, vp)
                for t in fixation_on_terms:
                    ret[t] += fixation_on_terms[t]
        if terms:
            return {t: ret[t] for t in ret if t in terms}
        else:
            return dict(ret)


class FixationTimeAnalyzer(FixationTimeAnalyzerBase):

    def __init__(self, user_list=None):

        super(FixationTimeAnalyzer, self).__init__(user_list=user_list)
        self.extract_functions.append(('query_tf', self.query_term_tf))
        self.extract_functions.append(('non_query_tf', self.non_query_term_tf))

        self.extract_functions.append(('avg_f_time_query', self.avg_fixation_time_on_query_terms))
        self.extract_functions.append(('avg_f_time_non_query', self.avg_fixation_time_on_non_query_terms))
        self.extract_functions.append(('avg_f_time', self.avg_fixation_time))

        self.extract_functions.append(('avg_f_time_query_serp',
                                       self.avg_fixation_time_on_query_terms_on_serp))
        self.extract_functions.append(('avg_f_time_non_query_serp',
                                       self.avg_fixation_time_on_non_query_terms_on_serp))

        self.extract_functions.append(('avg_f_time_query_landing_page',
                                       self.avg_fixation_time_on_query_terms_on_landing_page))
        self.extract_functions.append(('avg_f_time_non_query_landing_page',
                                       self.avg_fixation_time_on_non_query_terms_on_landing_page))

    def _query_term_tf(self, row, task_session, page_type=None):
        query_terms = set()
        for q in task_session.queries:
            terms = self._word_segment(q.query)
            for t in terms:
                print t + ' ',
            query_terms.update(terms)
            print
        print 'Query terms:',
        for q in query_terms:
            print q + ' ',

        pages = self._get_all_pages_in_task_session(task_session)
        if page_type:
            pages = [p for p in pages if isinstance(p, page_type)]
        documents = [self._get_text_from_page(p) for p in pages]

        tf = self._get_term_freq(documents, terms=query_terms)
        total_tf = np.sum(tf.values())
        return total_tf

    def query_term_tf(self, row, task_session):
        return self._query_term_tf(row, task_session)

    def non_query_term_tf(self, row, task_session):
        return self._non_query_term_tf(row, task_session)

    def _non_query_term_tf(self, row, task_session, page_type=None):
        query_terms = set()
        for q in task_session.queries:
            query_terms.update(self._word_segment(q.query))

        pages = self._get_all_pages_in_task_session(task_session)
        if page_type:
            pages = [p for p in pages if isinstance(p, page_type)]
        documents = [self._get_text_from_page(p) for p in pages]

        tf = self._get_term_freq(documents)
        non_query_terms = set(tf.keys()) - query_terms
        tf = {t : tf[t] for t in tf if t in non_query_terms}

        total_tf = np.sum(tf.values())
        return total_tf

    def _avg_fixation_time_on_query_terms(self, row, task_session, page_type=None):
        query_terms = set()
        for q in task_session.queries:
            query_terms.update(self._word_segment(q.query))

        try:
            if page_type:
                total_tf = row['query_term_tf_'+page_type.__name__]
            else:
                total_tf = row['query_term_tf']
        except KeyError:
            total_tf = self._query_term_tf(row, task_session, page_type=page_type)

        fixation_time = self._get_fixation_time(task_session, terms=query_terms)
        total_fixation_time = np.sum(fixation_time.values())
        return 1.0 * total_fixation_time / total_tf

    def _avg_fixation_time_on_non_query_terms(self, row, task_session, page_type=None):
        query_terms = set()
        for q in task_session.queries:
            query_terms.update(self._word_segment(q.query))

        pages = self._get_all_pages_in_task_session(task_session)
        if page_type:
            pages = [p for p in pages if isinstance(p, page_type)]
        documents = [self._get_text_from_page(p) for p in pages]

        tf = self._get_term_freq(documents)
        non_query_terms = set(tf.keys()) - query_terms
        tf = {t : tf[t] for t in tf if t in non_query_terms}

        total_tf = np.sum(tf.values())
        fixation_time = self._get_fixation_time(task_session, terms=non_query_terms)
        total_fixation_time = np.sum(fixation_time.values())
        return 1.0 * total_fixation_time / total_tf

    def avg_fixation_time_on_query_terms(self, row, task_session):
        return self._avg_fixation_time_on_query_terms(row, task_session)

    def avg_fixation_time_on_non_query_terms(self, row, task_session):
        return self._avg_fixation_time_on_non_query_terms(row, task_session)

    def avg_fixation_time_on_query_terms_on_serp(self, row, task_session):
        return self._avg_fixation_time_on_query_terms(row, task_session, page_type=SERPPage)

    def avg_fixation_time_on_non_query_terms_on_serp(self, row, task_session):
        return self._avg_fixation_time_on_non_query_terms(row, task_session, page_type=SERPPage)

    def avg_fixation_time_on_query_terms_on_landing_page(self, row, task_session):
        return self._avg_fixation_time_on_query_terms(row, task_session, page_type=LandingPage)

    def avg_fixation_time_on_non_query_terms_on_landing_page(self, row, task_session):
        return self._avg_fixation_time_on_non_query_terms(row, task_session, page_type=LandingPage)

    def avg_fixation_time(self, row, task_session):
        return (row.query_tf*row.avg_f_time_query + row.non_query_tf*row.avg_f_time_non_query) \
               / (row.query_tf + row.non_query_tf)


class TermSourceAnalyzer(FixationTimeAnalyzerBase):

    def __init__(self, user_list=None):
        super(TermSourceAnalyzer, self).__init__(user_list=user_list)

        self.extract_functions.append(('new_term_with_out_fixation', self.term_without_fixation))
        self.extract_functions.append(('new_term_with_out_fixation100', self.term_without_fixation100))
        self.extract_functions.append(('new_term_with_out_fixation200', self.term_without_fixation200))
        self.extract_functions.append(('new_term_with_out_fixation500', self.term_without_fixation500))

        self.extract_functions.append(('new_term_ratio', self.new_term_ratio))
        self.extract_functions.append(('new_term_from_serp', self.new_term_from_serp))
        self.extract_functions.append(('new_term_from_landing_page', self.new_term_from_landing_page))
        self.extract_functions.append(('new_term_from_fixation', self.new_term_from_fixation))
        self.extract_functions.append(('new_term_from_fixation_on_serp', self.new_term_from_fixation_on_serp))
        self.extract_functions.append(('new_term_from_fixation_on_landing_page',
                                       self.new_term_from_fixation_on_landing_page))

        self.extract_functions.append(('query_term_from_desc', self.query_term_from_desc))
        self.extract_functions.append(('query_term_from_serp', self.query_term_from_serp))
        self.extract_functions.append(('query_term_from_landing_page', self.query_term_from_landing_page))
        self.extract_functions.append(('query_term_from_fixation', self.query_term_from_fixation))
        self.extract_functions.append(('query_term_from_fixation_on_serp', self.query_term_from_fixation_on_serp))
        self.extract_functions.append(('query_term_from_fixation_on_landing_page',
                                       self.query_term_from_fixation_on_landing_page))

        self.extract_functions.append(('add_term_ratio', self.add_term_ratio))
        self.extract_functions.append(('add_term_from_desc', self.add_term_from_desc))
        self.extract_functions.append(('add_term_from_serp', self.add_term_from_serp))
        self.extract_functions.append(('add_term_from_landing_page', self.add_term_from_landing_page))
        self.extract_functions.append(('add_term_from_fixation', self.add_term_from_fixation))
        self.extract_functions.append(('add_term_from_fixation_on_serp', self.add_term_from_fixation_on_serp))
        self.extract_functions.append(('add_term_from_fixation_on_landing_page',
                                       self.add_term_from_fixation_on_landing_page))

    def term_without_source(self, row, task_session):
        task_id = task_session.task_url.split('/')[2]
        task = get_task_by_id(task_id)
        terms_from_source = set(self._word_segment(task.description))
        query_and_term_list = self._get_query_and_new_terms(task_session)
        num_q_terms = 0.0
        num_terms_without_source = 0.0

        pages = [p for p in self._get_all_pages_in_task_session(task_session)]
        for q, terms in query_and_term_list:
            for p in pages:
                if p.start_time > q.start_time:
                    continue
                terms_from_source.update(self._word_segment(self._get_text_from_page(p)))
            num_q_terms += len(terms)
            num_terms_without_source += len([t for t in terms if t not in terms_from_source])

        return num_terms_without_source

    def _get_new_terms_without_fixation(self, task_session, threshold=0):
        task_id = task_session.task_url.split('/')[2]
        task = get_task_by_id(task_id)
        terms_from_source = set(self._word_segment(task.description))
        query_and_term_list = self._get_query_and_new_terms(task_session)
        ret = []

        pages = [p for p in self._get_all_pages_in_task_session(task_session)]
        for q, terms in query_and_term_list:
            for p in pages:
                if p.start_time > q.start_time:
                    continue
                for vp in p.viewports:
                    if vp.start_time > q.start_time:
                        continue
                    fixation_on_terms = self._get_viewport_text(p, vp)
                    terms_from_source.update(
                        [t for t in fixation_on_terms if fixation_on_terms[t] > threshold]
                    )
            ret += [t for t in terms if t not in terms_from_source]

        return ret

    def _term_without_fixation(self, row, task_session, threshold=0):
        task_id = task_session.task_url.split('/')[2]
        task = get_task_by_id(task_id)
        terms_from_source = set(self._word_segment(task.description))
        query_and_term_list = self._get_query_and_new_terms(task_session)

        new_terms_without_fixation = self._get_new_terms_without_fixation(row, task_session, threshold=threshold)

        num_new_terms = 0.1 + sum([len(terms) for q, terms in query_and_term_list])
        num_new_terms_without_fixation = sum(len(terms) for terms in new_terms_without_fixation)

        '''
        pages = [p for p in self._get_all_pages_in_task_session(task_session)]
        for q, terms in query_and_term_list:
            for p in pages:
                if p.start_time > q.start_time:
                    continue
                for vp in p.viewports:
                    if vp.start_time > q.start_time:
                        continue
                    fixation_on_terms = self._get_viewport_text(p, vp)
                    terms_from_source.update(
                        [t for t in fixation_on_terms if fixation_on_terms[t] > threshold]
                    )
            num_q_terms += len(terms)
            num_terms_without_source += len([t for t in terms if t not in terms_from_source])
        '''

        return num_new_terms_without_fixation / num_new_terms

    def term_without_fixation(self, row, task_session):
        return self._term_without_fixation(row, task_session)

    def term_without_fixation100(self, row, task_session):
        return self._term_without_fixation(row, task_session, threshold=100)

    def term_without_fixation200(self, row, task_session):
        return self._term_without_fixation(row, task_session, threshold=200)

    def term_without_fixation500(self, row, task_session):
        return self._term_without_fixation(row, task_session, threshold=500)

    def new_term_ratio(self, row, task_session):
        num_q_terms = sum([len(terms) for q, terms in self._get_query_and_terms(task_session)])
        num_new_terms = sum([len(terms) for q, terms in self._get_query_and_new_terms(task_session)])
        return 1.0 * num_new_terms / num_q_terms

    def add_term_ratio(self, row, task_session):
        num_q_terms = sum([len(terms) for q, terms in self._get_query_and_terms(task_session)])
        num_add_terms = sum([len(terms) for q, terms in self._get_query_and_add_terms(task_session)])
        return 1.0 * num_add_terms / num_q_terms

    @staticmethod
    def _term_from_source(terms_list, source_list):
        num_term_from_source = 0.0
        num_terms = 0.0
        for terms, source_terms in zip(terms_list, source_list):
            for t in terms:
                num_terms += 1.0
                if t in source_terms:
                    num_term_from_source += 1.0
        if num_terms == 0.0:
            return 1.0
        else:
            return num_term_from_source / num_terms

    def _term_from_description(self, row, task_session, extract_terms=None):
        task_id = task_session.task_url.split('/')[2]
        task = get_task_by_id(task_id)
        task_desc_terms = set(self._word_segment(task.description))
        query_and_term_list = extract_terms(task_session)
        return self._term_from_source([terms for _, terms in query_and_term_list],
                                          [task_desc_terms] * len(query_and_term_list))

    def _term_from_serp(self, row, task_session, extract_terms=None):
        query_and_term_list = extract_terms(task_session)
        queries = [x[0] for x in query_and_term_list]
        terms_list = [x[1] for x in query_and_term_list]

        source_list = []
        for idx, query in enumerate(queries):
            serp_terms = set()
            if idx > 0:
                for p in query.pages:
                    serp_terms.update(self._word_segment(self._get_text_from_page(p)))
            source_list.append(serp_terms)
        return self._term_from_source(terms_list, source_list)

    def _term_from_landing_page(self, row, task_session, extract_terms=None):
        pages = [p for p in self._get_all_pages_in_task_session(task_session) if isinstance(p, LandingPage)]
        query_and_term_list = extract_terms(task_session)
        queries = [x[0] for x in query_and_term_list]
        terms_list = [x[1] for x in query_and_term_list]

        source_list = []
        for idx, q in enumerate(queries):
            landing_page_terms = set()
            if idx > 0:
                for p in pages:
                    if p.start_time >= q.start_time or p.end_time <= queries[idx-1].start_time:
                        continue
                    landing_page_terms.update(self._word_segment(self._get_text_from_page(p)))
            source_list.append(landing_page_terms)

        return self._term_from_source(terms_list, source_list)

    def _term_from_fixated_elements(self, row, task_session, extract_terms=None, page_type=None):
        pages = [p for p in self._get_all_pages_in_task_session(task_session)]
        if page_type:
            pages = [p for p in pages if isinstance(p, page_type)]

        query_and_term_list = extract_terms(task_session)
        queries = [x[0] for x in query_and_term_list]
        terms_list = [x[1] for x in query_and_term_list]
        source_list = []
        for idx, q in enumerate(queries):
            fixated_terms = set()
            if idx > 0:
                for p in pages:
                    for vp in p.viewports:
                        if vp.start_time >= q.start_time or vp.end_time <= queries[idx-1].start_time:
                            continue
                        fixated_terms.update(self._get_viewport_text(p, vp).keys())
            source_list.append(fixated_terms)

        return self._term_from_source(terms_list, source_list)

    def query_term_from_desc(self, row, task_session):
        return self._term_from_description(row, task_session, extract_terms=self._get_query_and_terms)

    def query_term_from_serp(self, row, task_session):
        return self._term_from_serp(row, task_session, extract_terms=self._get_query_and_terms)

    def query_term_from_landing_page(self, row, task_session):
        return self._term_from_landing_page(row, task_session, extract_terms=self._get_query_and_terms)

    def query_term_from_fixation(self, row, task_session):
        return self._term_from_fixated_elements(row, task_session, extract_terms=self._get_query_and_terms)

    def query_term_from_fixation_on_serp(self, row, task_session):
        return self._term_from_fixated_elements(row, task_session,
                                                extract_terms=self._get_query_and_terms,
                                                page_type=SERPPage)

    def query_term_from_fixation_on_landing_page(self, row, task_session):
        return self._term_from_fixated_elements(row, task_session,
                                                extract_terms=self._get_query_and_terms,
                                                page_type=LandingPage)

    def add_term_from_desc(self, row, task_session):
        return self._term_from_description(row, task_session, extract_terms=self._get_query_and_add_terms)

    def add_term_from_serp(self, row, task_session):
        return self._term_from_serp(row, task_session, extract_terms=self._get_query_and_add_terms)

    def add_term_from_landing_page(self, row, task_session):
        return self._term_from_landing_page(row, task_session, extract_terms=self._get_query_and_add_terms)

    def add_term_from_fixation(self, row, task_session):
        return self._term_from_fixated_elements(row, task_session, extract_terms=self._get_query_and_add_terms)

    def add_term_from_fixation_on_serp(self, row, task_session):
        return self._term_from_fixated_elements(row, task_session,
                                                extract_terms=self._get_query_and_add_terms,
                                                page_type=SERPPage)

    def add_term_from_fixation_on_landing_page(self, row, task_session):
        return self._term_from_fixated_elements(row, task_session,
                                                extract_terms=self._get_query_and_add_terms,
                                                page_type=LandingPage)

    def new_term_from_serp(self, row, task_session):
        return self._term_from_serp(row, task_session, extract_terms=self._get_query_and_new_terms)

    def new_term_from_landing_page(self, row, task_session):
        return self._term_from_landing_page(row, task_session, extract_terms=self._get_query_and_new_terms)

    def new_term_from_fixation(self, row, task_session):
        return self._term_from_fixated_elements(row, task_session, extract_terms=self._get_query_and_new_terms)

    def new_term_from_fixation_on_serp(self, row, task_session):
        return self._term_from_fixated_elements(row, task_session,
                                                extract_terms=self._get_query_and_new_terms,
                                                page_type=SERPPage)

    def new_term_from_fixation_on_landing_page(self, row, task_session):
        return self._term_from_fixated_elements(row, task_session,
                                                extract_terms=self._get_query_and_new_terms,
                                                page_type=LandingPage)

    def output_new_terms(self, threshold=0):

        columns = ['task_desc',
                   'in_domain_new_terms', 'out_domain_new_terms',
                   'in_domain_new_terms_without_f', 'out_domain_without_f']
        data = []
        for task_id in range(1, 7):
            task = get_task_by_id(str(task_id))

            indicator = self.session_df.task_id == task_id
            in_domain_ind = indicator & (self.session_df.in_domain == 1)
            out_domain_ind = indicator & (self.session_df.in_domain == 0)

            in_domain_new_terms = Counter()
            in_domain_new_terms_without_f = Counter()
            for ts in self._task_session_stat(in_domain_ind):
                q_t_list = self._get_query_and_new_terms(ts)
                for q, terms in q_t_list:
                    in_domain_new_terms.update(terms)
                terms_without_f = self._get_new_terms_without_fixation(ts, threshold=threshold)
                in_domain_new_terms_without_f.update(terms_without_f)

            out_domain_new_terms = Counter()
            out_domain_new_terms_without_f = Counter()
            for ts in self._task_session_stat(out_domain_ind):
                q_t_list = self._get_query_and_new_terms(ts)
                for q, terms in q_t_list:
                    out_domain_new_terms.update(terms)
                terms_without_f = self._get_new_terms_without_fixation(ts, threshold=threshold)
                out_domain_new_terms_without_f.update(terms_without_f)

            data.append(
                [task.description,
                 ]
            )

        return columns, data


def output_new_terms(threshold):
    a = TermSourceAnalyzer()
    return a.output_new_terms(threshold=threshold)

def test():
    a = TermSourceAnalyzer()
    return a.get_data_df()

if __name__ == '__main__':
    '''
    a = FixationTimeAnalyzer()
    a.check_connection()
    df = a.get_data_df()
    df.to_pickle(ROOT + '/tmp/fixation_time_new.dataframe')
    '''

    a = TermSourceAnalyzer()
    a.check_connection()
    df = a.get_data_df()
    df.to_pickle(ROOT + '/tmp/term_source_new.dataframe')

    print DataAnalyzer.dataframe_stat(df)


