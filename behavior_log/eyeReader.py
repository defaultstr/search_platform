__author__ = 'defaultstr'

import time


class EyeReader(object):
    def __init__(self):
        self.fin = None
        pass

    def open(self, filename):
        self.fin = open(filename, 'r')
        self.col_names = self.fin.readline().rstrip().split('\t')
        self.col_name_to_idx = {self.col_names[i]: i for i in range(0, len(self.col_names))}
        return self.fin

    def get_by_name(self, line, name):
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
            return None

    def click_in_rect(self, line, rect):
        if self.get_by_name(line, 'MouseEvent') != 'Left':
            return False
        try:
            x = int(self.get_by_name(line, 'MouseEventX (MCSpx)'))
            y = int(self.get_by_name(line, 'MouseEventY (MCSpx)'))
            if ((x >= rect[0]) and (x <= rect[2])
                and (y >= rect[1]) and (y <= rect[3])):
                #print x, y
                return True
            else:
                return False
        except ValueError:
            return False

    def get_time_stamp(self, line):
        recording_date = self.get_by_name(line, 'RecordingDate')
        local_time_str = self.get_by_name(line, 'LocalTimeStamp')
        t = time.strptime(recording_date+' '+local_time_str.split('.')[0], '%d/%m/%Y %H:%M:%S')
        timestamp = time.mktime(t) - time.mktime((1970, 1, 1, 0, 0, 0, 0, 0, 0)) - 8 * 3600
        timestamp = int(timestamp * 1000 + int(local_time_str.split('.')[1]))
        return timestamp

    @staticmethod
    def time_stamp_to_local_time_str(timestamp):
        sec = timestamp / 1000
        milli = timestamp % 1000
        return time.strftime('%d/%m/%Y %H:%M:%S', time.localtime(sec))+'.'+str(milli)


