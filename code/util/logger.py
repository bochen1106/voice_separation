#!/usr/bin/python
import os
import time
import datetime


class Logger(object):


    def __init__(self, filename_log):

        self.file = None
        if filename_log is not None:
            log_dir = os.path.split(filename_log)[0]
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            self.file = open(filename_log, 'wt')
        self.verbose_level = 2

    def log(self, contents, level=1):
        # print level below verbose level
        st = datetime.datetime.fromtimestamp(time.time()).strftime('[%Y-%m-%d %H:%M:%S]')
        if level <= self.verbose_level:
            line = contents
            if isinstance(contents, list):
                line = '\n'.join(line)
                line = st + '\n' + line
            else:
                line = st + ': ' + line

            print(line)

            line = line + '\n'
            if self.file is not None:
                self.file.writelines(line)
                self.file.flush()

