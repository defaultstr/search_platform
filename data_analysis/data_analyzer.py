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
from mongoengine import *
from behavior_log.models import *
from user_system.models import *
import jieba


def variable_decorator(func):
    return func

def query_var_decorator(func):
    return func


class DataAnalyzer(object):

    def __init__(self, user_list=None):
        df = pd.read_csv('./behavior_log/userlist.tsv', sep='\t')
        self.user_list = list(set(df.username))
        if user_list:
            self.user_list = [u for u in self.user_list if u in user_list]

        self.session_df = None
        self.task_sessions = []
        self.init_session_df(df)

        self.query_df = None
        self.init_query_df()

        self.extract_functions = []
        self.extract_function_dec_string = self.find_decorators(target=DataAnalyzer)['dec_test'][0]

        self.query_extract_functions = []
        self.query_extract_function_dec_string = self.find_decorators(target=DataAnalyzer)['query_dec_test'][0]

        self.get_extract_functions()

    def init_session_df(self, df):
        columns = ['uid', 'user_domain', 'task_id', 'task_domain', 'in_domain']
        data = []
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

    def init_query_df(self):
        columns = ['uid', 'user_domain', 'task_id', 'task_domain', 'in_domain', 'session_idx', 'query_idx']
        data = []
        for idx, row in self.session_df.iterrows():
            task_session = self.task_sessions[idx]
            assert isinstance(task_session, TaskSession)
            row = list(row[0:5]) + [idx]
            for q_idx in range(len(task_session.queries)):
                data.append(row + [q_idx])
        self.query_df = pd.DataFrame(columns=columns, data=data)

    @variable_decorator
    def dec_test(self):
        pass

    @query_var_decorator
    def query_dec_test(self):
        pass

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

    def iter_query_sessions(self):
        for idx, row in self.query_df.iterrows():
            yield idx, row, \
                  self.task_sessions[row.session_idx], \
                  self.task_sessions[row.session_idx].queries[row.query_idx]

    def add_query_variable(self,
                           variable_name='page_num',
                           extract_function=lambda row, task_session, query: len(query.pages)):
        data = []
        for idx, row, task_session, query in self.iter_query_sessions():
            print row.uid, row.task_id, row.query_idx
            data.append(extract_function(row, task_session, query))
            print '%s=%f' % (variable_name, data[-1])
        self.query_df[variable_name] = data

    def get_query_df(self):
        return self.query_df

    def get_query_data_df(self):
        for name, extract_func in self.query_extract_functions:
            print 'Extracting Query Variable: %s' % name
            self.add_query_variable(variable_name=name, extract_function=extract_func)
            print 'Finish Extracting Variable: %s' % name
            self.query_df.to_pickle('./tmp/tmp_query.dataframe')
        return self.query_df

    def _task_session_stat(self, indicate_series=None):
        if indicate_series is not None:
            ret = []
            assert(len(indicate_series) == len(self.session_df) == len(self.task_sessions))
            for idx, ind in enumerate(indicate_series):
                if ind:
                    ret.append(self.task_sessions[idx])
            return ret
        else:
            return list(self.task_sessions)


    @staticmethod
    def _word_segment(text):
        """

        :param text: a text to segment
        :return: a list of segmented terms
        """
        return [w for w in jieba.cut(text) if w.strip() != '']

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
        from scipy.stats import ks_2samp

        columns = ['variable',
                   'mean',
                   'mean_in_domain',
                   'mean_out_of_domain',
                   'diff',
                   'F', 'p_anova',
                   'U', 'p_mannwhitenyU',
                   'D', 'p_ks']

        data = []
        in_df = df[df.in_domain == 1]
        out_df = df[df.in_domain == 0]
        info_idx = 5
        if 'query_idx' in df.columns:
            info_idx = 6
        for x in df.columns[info_idx:]:
            F, p_anova = f_oneway(in_df[x], out_df[x])
            U, p_mann = mannwhitneyu(in_df[x], out_df[x])
            D, p_ks = ks_2samp(in_df[x], out_df[x])
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
                U, p_mann,
                D, p_ks,
            ]
            data.append(row)

        return pd.DataFrame(columns=columns, data=data)

    @staticmethod
    def get_user_normalized_dataframe(df):
        v_df = df[['uid']+list(df.columns[5:])]
        nv_df = v_df.groupby('uid').transform(lambda x: (x - x.mean()) / x.std()).fillna(value=0.0)
        df = pd.concat([df[df.columns[0:5]], nv_df], axis=1)
        return df


    @staticmethod
    def dataframe_stat_task_normalized(df):
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

        v_df = df[['task_id']+list(df.columns[5:])]
        nv_df = v_df.groupby('task_id').transform(lambda x: (x - x.mean()) / x.std())
        df = pd.concat([df[df.columns[0:5]], nv_df], axis=1)

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

    def find_decorators(self, target=None):
        import ast, inspect
        if target is None:
            target = self.__class__
        res = {}

        def visit_function_def(node):
            res[node.name] = [ast.dump(e) for e in node.decorator_list]

        visitor = ast.NodeVisitor()
        visitor.visit_FunctionDef = visit_function_def
        visitor.visit(compile(inspect.getsource(target), '?', 'exec', ast.PyCF_ONLY_AST))
        #print res
        return res

    def get_extract_functions(self):
        res = self.find_decorators()

        for key in sorted(res.keys()):
            if self.extract_function_dec_string in res[key]:
                self.extract_functions.append((key, getattr(self, key)))
            if self.query_extract_function_dec_string in res[key]:
                self.query_extract_functions.append((key, getattr(self, key)))

        print 'session variables:'
        for name, _ in self.extract_functions:
            print '\t'+name,
        print '\nquery variables:'
        for name, _ in self.query_extract_functions:
            print '\t'+name,


if __name__ == '__main__':
    a = DataAnalyzer()
    print a.find_decorators()

