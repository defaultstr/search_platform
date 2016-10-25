#!/user/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

from data_analyzer import DataAnalyzer
from mongoengine import connect
from behavior_log.models import *
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
import jieba
import json
import numpy as np
from exp_domain_expertise.utils import get_task_by_id


class FixationTimeAnalyzerBase(DataAnalyzer):

    def __init__(self, user_list=None):
        super(FixationTimeAnalyzerBase, self).__init__(user_list=user_list)

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

        self.extract_functions.append(('query_term_from_desc', self.query_term_from_desc))
        self.extract_functions.append(('query_term_from_serp', self.query_term_from_serp))
        self.extract_functions.append(('query_term_from_landing_page', self.query_term_from_landing_page))
        self.extract_functions.append(('query_term_from_fixation', self.query_term_from_fixation))
        self.extract_functions.append(('query_term_from_fixation_on_serp', self.query_term_from_fixation_on_serp))
        self.extract_functions.append(('query_term_from_fixation_on_landing_page',
                                       self.query_term_from_fixation_on_landing_page))

        self.extract_functions.append(('add_term_from_desc', self.add_term_from_desc))
        self.extract_functions.append(('add_term_from_serp', self.add_term_from_serp))
        self.extract_functions.append(('add_term_from_landing_page', self.add_term_from_landing_page))
        self.extract_functions.append(('add_term_from_fixation', self.add_term_from_fixation))
        self.extract_functions.append(('add_term_from_fixation_on_serp', self.add_term_from_fixation_on_serp))
        self.extract_functions.append(('add_term_from_fixation_on_landing_page',
                                       self.add_term_from_fixation_on_landing_page))

    def _get_query_and_add_terms(self, task_session):
        ret = []
        used_terms = set()
        for q in task_session.queries:
            query_terms = self._word_segment(q.query)
            add_terms = list(set(query_terms) - used_terms)
            used_terms.update(add_terms)
            ret.append((q, add_terms))
        return ret

    def _get_query_and_terms(self, task_session):
        return [(q, list(set(self._word_segment(q.query)))) for q in task_session.queries]

    @staticmethod
    def _term_from_source(add_terms_list, source_list):
        num_add_term_from_desc = 0.0
        num_add_terms = 0.0
        for add_terms, source_terms in zip(add_terms_list, source_list):
            for t in add_terms:
                num_add_terms += 1.0
                if t in source_terms:
                    num_add_term_from_desc += 1.0
        if num_add_terms == 0.0:
            return 1.0
        else:
            return num_add_term_from_desc / num_add_terms

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


def test():
    a = TermSourceAnalyzer()
    a.check_connection()
    return a.get_data_df()


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../')
    df = test()
    df.to_pickle('./tmp/term_source.dataframe')
