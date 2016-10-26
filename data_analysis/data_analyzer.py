#!/user/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

import pandas as pd
from mongoengine import *
from behavior_log.models import *
from user_system.models import *


class DataAnalyzer(object):

    def __init__(self, user_list=None):
        df = pd.read_csv('./behavior_log/userlist.tsv', sep='\t')
        self.user_list = list(set(df.username))
        if user_list:
            self.user_list = [u for u in self.user_list if u in user_list]

        columns = ['uid', 'user_domain', 'task_id', 'task_domain', 'in_domain']
        data = []
        self.task_sessions = []
        for user in self.user_list:
            for task_id in range(1, 7):
                row = []
                row.append(user)
                user_domain = df[df.username == user].iloc[0].domain
                row.append(user_domain)

                try:
                    user_doc = User.objects.get(username=user)
                    task_url = '/exp_domain_expertise/%d/' % task_id
                    task_session_doc = TaskSession.objects.get(user=user_doc, task_url=task_url)
                    self.task_sessions.append(task_session_doc)
                except DoesNotExist:
                    continue

                row.append(task_id)
                task_domain = (task_id - 1) / 2
                row.append(task_domain)
                row.append(1 if user_domain == task_domain else 0)

                data.append(row)
        self.session_df = pd.DataFrame(columns=columns, data=data)
        self.extract_functions = []

    def iter_task_sessions(self):
        for idx, row in self.session_df.iterrows():
            yield idx, row, self.task_sessions[idx]

    def add_variable(self,
                     variable_name='query_num',
                     extract_function=lambda row, task_session: len(task_session.queries)):
        data = []
        for idx, row, task_session in self.iter_task_sessions():
            print row.uid, row.task_id
            data.append(extract_function(row, task_session))
            print '%s=%f' % (variable_name, data[-1])
        self.session_df[variable_name] = data

    def get_session_df(self):
        return self.session_df

    def get_data_df(self):
        for name, extract_function in self.extract_functions:
            print 'Extracting Variable: %s' % name
            self.add_variable(variable_name=name, extract_function=extract_function)
            print 'Finished Extracting Variable: %s' % name
            self.session_df.to_pickle('./tmp/tmp.dataframe')
        return self.session_df

    def _task_session_stat(self, indicate_series=None):
        if indicate_series:
            ret = []
            assert(len(indicate_series) == len(self.session_df) == len(self.task_sessions))
            for idx, ind in enumerate(indicate_series):
                if ind:
                    ret.append(self.task_sessions[idx])
            return ret
        else:
            return list(self.task_sessions)

    @staticmethod
    def _get_all_pages(page):
        ret = [page]
        for p in page.clicked_pages:
            ret += DataAnalyzer._get_all_pages(p)
        return ret

    @staticmethod
    def _get_all_pages_in_task_session(task_session):
        ret = []
        for q in task_session.queries:
            for p in q.pages:
                ret += DataAnalyzer._get_all_pages(p)
        return sorted(ret, key=lambda p: p.start_time)

    @staticmethod
    def check_connection():
        try:
            task_session = TaskSession.objects()
        except ConnectionError:
            connect('search_platform')

    @staticmethod
    def dataframe_stat(df):
        import numpy as np
        from scipy.stats import f_oneway
        from scipy.stats import mannwhitneyu
        columns = ['variable',
                   'mean',
                   'mean_in_domain',
                   'mean_out_of_domain',
                   'diff',
                   'F', 'p_anova',
                   'U', 'p_mannwhitenyU']

        data = []
        in_df = df[df.in_domain == 1]
        out_df = df[df.in_domain == 0]
        for x in df.columns[5:]:
            F, p_anova = f_oneway(in_df[x], out_df[x])
            U, p_mann = mannwhitneyu(in_df[x], out_df[x])
            row = [
                x,
                np.mean(df[x]),
                #np.std(df[x]),
                np.mean(in_df[x]),
                #np.std(in_df[x]),
                np.mean(out_df[x]),
                #np.std(out_df[x]),
                (np.mean(in_df[x]) - np.mean(out_df[x])) / np.mean(out_df[x]),
                F, p_anova,
                U, p_mann
            ]
            data.append(row)

        return pd.DataFrame(columns=columns, data=data)



