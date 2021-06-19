# -*- coding: UTF-8 -*-

import sys
import time


class ShowProcess():
    """
    ShowProcess
    """
    i = 0
    max_steps = 0
    max_arrow = 25
    infoDone = 'done'

    def __init__(self, max_steps, info='', infoDone='Done'):
        self.info = info
        self.max_steps = max_steps
        self.i = 0
        self.infoDone = infoDone

    # [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, i=None, showInfo='', restTime=''):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow
        percent = self.i * 100.0 / self.max_steps
        infoDone = ', ' + self.infoDone if self.i >= self.max_steps else ''
        if restTime != '':
            restTime = '(time remaining: %.3f sec)' % restTime

        process_bar = '[' + self.info + '] [' + '>' * num_arrow  \
            + '-' * num_line + ']'                               \
            + ' %d / %d, ' % (self.i, self.max_steps)            \
            + '%.2f' % percent + '%' + ' '                       \
            + showInfo + ' ' + restTime + infoDone               \
            + '\r'
        sys.stdout.write(process_bar)
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('')
        # print(self.infoDone)
        self.i = 0


if __name__ == '__main__':
    max_steps = 50

    process_bar = ShowProcess(max_steps, 'OK')

    for i in range(max_steps):
        process_bar.show_process(i+1)
        time.sleep(0.01)
    time.sleep(50)
