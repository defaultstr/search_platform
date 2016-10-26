#!/user/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'


import pandas as pd
import time
import numpy as np


class EyeReader(object):
    """
    Eye-tracking log reader
    example:
    r = EyeReader()
    r.open('eye-tracking-log.tsv')
    r.time_adjust(click_timestamps)
    fixations = r.get_fixation(start_time, end_time)
    """
    def __init__(self):
        self.fixation_df = None
        self.click_list = None
        self.fin = None
        self.col_names = None
        self.col_name_to_idx = None
        self.adjust_time = 0
        self.diff_local_recording = 0


    def _get_by_name(self, line, name):
        e = line.split('\t')
        if name in self.col_name_to_idx:
            '''
            print self.col_names
            print len(self.col_names)
            print e
            print len(e)
            print self.col_name_to_idx[name]
            '''
            return e[self.col_name_to_idx[name]]
        else:
            raise ValueError('The column name "%s" is not in this tsv file!' % name)

    def open(self, filename):
        self.fin = open(filename, 'r')
        self.col_names = self.fin.readline().rstrip().split('\t')
        self.col_name_to_idx = {self.col_names[i]: i for i in range(0, len(self.col_names))}

        # construct fixation df
        columns = ['timestamp', 'fixation_idx', 'duration',
                   'x_on_screen', 'y_on_screen']

        # compute the diff between LocalTimestamp and RecordindTimestamp

        line = self.fin.readline()
        local_timestamp = self.get_time_stamp(line)
        rec_timestamp = int(self._get_by_name(line, 'RecordingTimestamp'))
        self.diff_local_recording = local_timestamp - rec_timestamp

        data = []
        cur_fixation_idx = -1
        for line in self.fin:
            timestamp = self.get_time_stamp(line)
            if self._get_by_name(line, 'GazeEventType') != 'Fixation':
                continue

            fixation_idx, duration = [int(self._get_by_name(line, name))
                                      for name in ['FixationIndex', 'GazeEventDuration']
                                      ]
            if fixation_idx == cur_fixation_idx:
                continue
            else:
                cur_fixation_idx = fixation_idx
            pos = [int(self._get_by_name(line, name))
                   for name in ['FixationPointX (MCSpx)', 'FixationPointY (MCSpx)']
                   if self._get_by_name(line, name) != '-'
                   ]
            if len(pos) != 2:
                continue
            x_on_screen, y_on_screen = pos

            data.append([
                timestamp, fixation_idx, duration,
                x_on_screen, y_on_screen
            ])
        self.fixation_df = pd.DataFrame(columns=columns, data=data)

        # construct click list for time adjust
        self.fin = open(filename, 'r')
        self.col_names = self.fin.readline().rstrip().split('\t')

        self.click_list = []
        for line in self.fin:
            if self._get_by_name(line, 'MouseEvent') != 'Left':
                continue
            self.click_list.append(self.get_time_stamp(line))

    def time_adjust(self, click_timestamps):
        # the basic assumption is that the timestamp logged
        # by javascript is * larger * than the timestamp logged
        # by Tobii studio because the event handling takes some
        # time
        j = len(self.click_list) - 1
        diff = []
        for i in range(len(click_timestamps) - 1, -1, -1):
            while j >= 0 and self.click_list[j] > click_timestamps[i]:
                j -= 1
            diff.append(click_timestamps[i] - self.click_list[j])
        diff = [x for x in diff if x < 500]
        if click_timestamps is None or len(diff) == 0:
            self.adjust_time = 0
        else:
            self.adjust_time = int(np.mean(diff))
        print 'adjust time is %d ms' % self.adjust_time

    def get_fixations(self, start_time=None, end_time=None):
        if self.fixation_df is None:
            raise ValueError('No opened fixation logs. Call open(filename) first!')

        res_df = self.fixation_df.copy()
        if start_time is not None:
            res_df = res_df[res_df.timestamp >= start_time - self.adjust_time]
        if end_time is not None:
            res_df = res_df[res_df.timestamp <= end_time - self.adjust_time]

        res_df['origin_timestamp'] = res_df['timestamp']
        res_df['timestamp'] += self.adjust_time

        return res_df

    def get_time_stamp(self, line):
        recording_date = self._get_by_name(line, 'RecordingDate')
        local_time_str = self._get_by_name(line, 'LocalTimeStamp')
        t = time.strptime(recording_date+' '+local_time_str.split('.')[0], '%Y/%m/%d %H:%M:%S')
        timestamp = time.mktime(t) - time.mktime((1970, 1, 1, 0, 0, 0, 0, 0, 0)) - 8 * 3600
        timestamp = int(timestamp * 1000 + int(local_time_str.split('.')[1]))
        return timestamp

    @staticmethod
    def timestamp_to_local_time_str(timestamp):
        sec = timestamp / 1000
        milli = timestamp % 1000
        return time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(sec))+'.'+str(milli)

    def get_recording_timestamp(self, mouse_timestamp):
        return mouse_timestamp - self.adjust_time - self.diff_local_recording

