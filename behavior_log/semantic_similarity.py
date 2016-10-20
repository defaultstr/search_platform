#!/user/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

import behavior_log.behavior_log_extractor as ble
from behavior_log.models import *
import numpy as np
import json
from bs4 import BeautifulSoup
from collections import defaultdict
from gensim import corpora, models, similarities
from gensim.matutils import cossim
import jieba
import pandas as pd

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class SemanticSimilarity(object):

    def __init__(self):

        self.all_viewports = None
        self.all_pages = None
        self.user_task_to_viewports = None
        self.dictionary = None
        self.tfidf = None


    @staticmethod
    def get_area(e):
        return (e['right'] - e['left']) * (e['bottom'] - e['top'])

    @staticmethod
    def fixation_on_elements(f, e, weight):
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
    def get_text_from_page(page):
        soup = BeautifulSoup(page.html)
        [s.extract() for s in soup('script')]
        [s.extract() for s in soup('style')]
        [s.extract() for s in soup('img')]
        return soup.text

    @staticmethod
    def word_segment(text):
        return [w for w in jieba.cut(text) if w.strip() != '']

    @staticmethod
    def get_all_queries():
        for username, task_url in ble.iter_user_and_task():
            queries = [q.query for q in Query.objects(username=username, task_url=task_url)]
        return queries

    def get_viewport_text(self, page, viewport):
        visible_elements = json.loads(page.visible_elements)
        sorted_elements = sorted(visible_elements, key=lambda e: self.get_area(e))
        fixation_time = np.zeros((len(visible_elements)))
        for f in viewport.fixations:
            weight = f.duration
            for idx, e in enumerate(sorted_elements):
                f_time = self.fixation_on_elements(f, e, weight)
                fixation_time[idx] += f_time
                weight -= f_time
                if weight <= 0:
                    break
        fixation_text = [
            (e['text'], fixation_time[idx])
            for idx, e in enumerate(sorted_elements) if fixation_time[idx] > 0
        ]
        all_text = [
            (self.get_text_from_page(page), float(viewport.end_time-viewport.start_time))
        ]
        return page, viewport, fixation_text, all_text

    def get_all_viewport_text(self):
        self.all_viewports = []
        self.user_task_to_viewports = defaultdict(list)
        for username, task_url in ble.iter_user_and_task():
            vp_test_list = self.get_viewport_text_for_task_session(username, task_url)
            self.all_viewports += vp_test_list
            self.user_task_to_viewports[(username, task_url)] = vp_test_list

    def get_viewport_text_for_task_session(self, username, task_url):
        ble.debug_print('Processing %s %s' % (username, task_url))
        ret = []
        # serp
        for serp in SERPPage.objects(username=username, task_url=task_url):
            for vp in serp.viewports:
                vp_text = self.get_viewport_text(serp, vp)
                ret.append(vp_text)
        # landing page
        for p in LandingPage.objects(username=username, task_url=task_url):
            for vp in p.viewports:
                vp_text = self.get_viewport_text(p, vp)
                ret.append(vp_text)
        return ret

    def train_semantic_model(self):
        texts = self.get_all_queries()
        self.get_all_viewport_text()
        for page, vp, fixation_text, all_text in self.all_viewports:
            texts += [text for text, _ in fixation_text]
            texts += [all_text[0][0]]
        texts = [self.word_segment(text) for text in texts]
        from collections import Counter
        c = Counter()
        for text in texts:
            c.update(text)
        texts = [[token for token in text if c[token] > 1] for text in texts]

        self.dictionary = corpora.Dictionary(texts)
        self.dictionary.save('./tmp/gensim.dict')
        print self.dictionary

        corpus = [self.dictionary.doc2bow(text) for text in texts]

        self.tfidf = models.TfidfModel(corpus)
        self.tfidf.save('./tmp/tfidf_model')
        corpus_tfidf = self.tfidf[corpus]

        self.lsi = models.LsiModel(corpus_tfidf, id2word=self.dictionary, num_topics=100)
        self.lsi.save('./tmp/lsi_model')
        corpus_lsi = self.lsi[corpus_tfidf]

    def similarity(self, text1, text2, model):
        text1 = self.word_segment(text1)
        text2 = self.word_segment(text2)
        bow1 = self.dictionary.doc2bow(text1)
        bow2 = self.dictionary.doc2bow(text2)
        v1 = model[bow1]
        v2 = model[bow2]
        return cossim(v1, v2)

    def compute_semantic_similarities(self, model):
        pages = []
        fixation_sims = []
        page_sims = []

        for username, task_url in ble.iter_user_and_task():
            queries = Query.objects(username=username, task_url=task_url)
            vp_text = self.user_task_to_viewports[(username, task_url)]
            vp_text = sorted(vp_text, key=lambda x: x[1].start_time)

            for q in queries:
                for page, vp, fixation_text, page_text in vp_text:

                    if vp.end_time > q.start_time:
                        break

                    if isinstance(page, SERPPage):
                        pages.append('SERP')
                    else:
                        pages.append('LandingPage')

                    time_diff = q.start_time - vp.end_time
                    f_sims = []
                    weights = []
                    for text, weight in fixation_text:
                        f_sims.append(self.similarity(q.query, text, model))
                        weights.append(weight)
                    f_sims = np.array(f_sims)
                    weights = np.array(weights)
                    total_duration = np.sum(weights)
                    weights /= total_duration

                    fixation_sims.append(
                        (
                            total_duration,
                            time_diff,
                            np.dot(f_sims, weights)
                        )
                    )

                    page_sims.append(
                        (
                            float(vp.end_time-vp.start_time),
                            time_diff,
                            self.similarity(q.query, page_text[0][0], model),
                        )
                    )

        print len(fixation_sims), len(page_sims), len(pages)
        return fixation_sims, page_sims, pages

    def compute_term_fixation_time(self):
        term_fixation_time = defaultdict(list)
        query_terms = set()

        for username, task_url in ble.iter_user_and_task():
            queries = Query.objects(username=username, task_url=task_url)
            vp_text = self.user_task_to_viewports[(username, task_url)]
            vp_text = sorted(vp_text, key=lambda x: x[1].start_time)

            for q in queries:
                query_terms.update(self.word_segment(q.query))
                for page, vp, fixation_text, page_text in vp_text:

                    if vp.end_time > q.start_time:
                        break
                    for text, weight in fixation_text:
                        terms = self.word_segment(text)
                        for t in terms:
                            term_fixation_time[t].append(weight / len(terms))

        ret = {}
        for term in term_fixation_time:
            ret[term] = np.mean(term_fixation_time[term])

        return ret, query_terms


    @staticmethod
    def save_similarities_to_dataframe(sims, pages, outputfile):
        columns = ['duration', 'time_diff', 'similarity']
        data = []
        df = pd.DataFrame(data=sims, columns=columns)
        df['page_type'] = pages
        df.to_csv(outputfile)


def test():
    ss = SemanticSimilarity()
    ss.get_all_viewport_text()
    term_fixation_time, query_terms = ss.compute_term_fixation_time()

    query_term_times = [term_fixation_time[t] for t in term_fixation_time if t in query_terms]
    print 'query term fixation:'
    print 'avg: %f\tstd: %f' % (np.mean(query_term_times), np.std(query_term_times))

    non_query_times = [term_fixation_time[t] for t in term_fixation_time if t not in query_terms]
    print 'non query term fixation:'
    print 'avg: %f\tstd: %f' % (np.mean(non_query_times), np.std(non_query_times))

    return query_term_times, non_query_times


def similarity():
    ss = SemanticSimilarity()
    ss.train_semantic_model()

    # tfidf
    fixation_sims, page_sims, pages = ss.compute_semantic_similarities(ss.tfidf)
    ss.save_similarities_to_dataframe(fixation_sims, pages, './tmp/tfidf_fixation.csv')
    ss.save_similarities_to_dataframe(page_sims, pages, './tmp/tfidf_page.csv')

    # LSI (d=100)
    fixation_sims, page_sims, pages = ss.compute_semantic_similarities(ss.lsi)
    ss.save_similarities_to_dataframe(fixation_sims, pages, './tmp/lsi100_fixation.csv')
    ss.save_similarities_to_dataframe(page_sims, pages, './tmp/lsi100_page.csv')






