#!/user/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

if __name__ == '__main__':
    import sys
    ROOT = sys.argv[1]
    sys.path.insert(0, ROOT)
    from mongoengine import *
    connect('search_platform')

import pandas as pd

from data_analyzer import DataAnalyzer, variable_decorator
from exp_domain_expertise.models import *


class FeedbackAnalyzer(DataAnalyzer):
    def __init__(self, user_list=None):
        super(FeedbackAnalyzer, self).__init__(user_list=user_list)

        score_df = pd.read_csv(ROOT + '/tmp/score.csv', index_col=0)
        self.answer_scores = {
            '%s_%d' % (row.uid, row.task_id): row.score
            for idx, row in score_df.iterrows()
            }

        '''
        self.extract_functions.append(('pre_knowledge', self.get_pre_log_knowledge))
        self.extract_functions.append(('pre_interest', self.get_pre_log_interest))
        self.extract_functions.append(('pre_difficulty', self.get_pre_log_difficulty))

        self.extract_functions.append(('post_knowledge', self.get_post_log_knowledge))
        self.extract_functions.append(('post_interest', self.get_post_log_interest))
        self.extract_functions.append(('post_difficulty', self.get_post_log_difficulty))
        self.extract_functions.append(('post_satisfaction', self.get_post_log_satisfaction))
        '''

    @variable_decorator
    def pre_log_knowledge(self, row, task_session):
        log = task_session.pre_task_question_log
        assert isinstance(log, PreTaskQuestionLog)
        return log.knowledge_scale

    @variable_decorator
    def pre_log_interest(self, row, task_session):
        log = task_session.pre_task_question_log
        assert isinstance(log, PreTaskQuestionLog)
        return log.interest_scale

    @variable_decorator
    def pre_log_difficulty(self, row, task_session):
        log = task_session.pre_task_question_log
        assert isinstance(log, PreTaskQuestionLog)
        return log.difficulty_scale

    @variable_decorator
    def post_log_knowledge(self, row, task_session):
        log = task_session.post_task_question_log
        assert isinstance(log, PostTaskQuestionLog)
        return log.knowledge_scale

    @variable_decorator
    def post_log_interest(self, row, task_session):
        log = task_session.post_task_question_log
        assert isinstance(log, PostTaskQuestionLog)
        return log.interest_scale

    @variable_decorator
    def post_log_difficulty(self, row, task_session):
        log = task_session.post_task_question_log
        assert isinstance(log, PostTaskQuestionLog)
        return log.difficulty_scale

    @variable_decorator
    def post_log_satisfaction(self, row, task_session):
        log = task_session.post_task_question_log
        assert isinstance(log, PostTaskQuestionLog)
        return log.satisfaction_scale

    @variable_decorator
    def answer_score(self, row, task_session):
        return self.answer_scores['%s_%d' % (row.uid, row.task_id)]

    def compute_other_session_features(self):
        '''
        self.session_df['delta_knowledge'] = self.session_df.post_log_knowledge - self.session_df.pre_log_knowledge
        self.session_df['delta_interest'] = self.session_df.post_log_interest - self.session_df.pre_log_interest
        self.session_df['delta_difficulty'] = self.session_df.post_log_difficulty - self.session_df.pre_log_difficulty
        '''
        pass

if __name__ == '__main__':
    a = FeedbackAnalyzer()
    a.check_connection()
    df = a.get_data_df()
    df.to_pickle(ROOT + '/tmp/feedback.dataframe')

    print DataAnalyzer.dataframe_stat(df)


