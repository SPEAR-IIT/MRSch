from datetime import datetime
import time
import re

import CqSim.bb_struc as Class_bb_struc

__metaclass__ = type


class bb_struc_SWF(Class_bb_struc.bb_struc):

    def bb_allocate(self, proc_num, job_index, start, end):
        # self.debug.debug("* "+self.myInfo+" -- bb_allocate",5)
        if self.is_available(proc_num) == 0:
            return 0
        self.idle -= proc_num
        self.avail = self.idle
        temp_job_info = {'job': job_index, 'end': end, 'bb': proc_num}
        j = 0
        is_done = 0
        temp_num = len(self.job_list)
        while (j < temp_num):
            if (temp_job_info['end'] < self.job_list[j]['end']):
                self.job_list.insert(j, temp_job_info)
                is_done = 1
                break
            j += 1

        if (is_done == 0):
            self.job_list.append(temp_job_info)
        '''
        self.debug.line(2,"...")
        for job in self.job_list:
            self.debug.debug(job['job'],2)
        self.debug.line(2,"...")
        '''
        self.debug.debug(
            "  Allocate" + "[" + str(job_index) + "]" + " Req:" + str(proc_num) + " Avail:" + str(self.avail) + " ", 4)
        return 1

    def bb_release(self, job_index, end):
        # self.debug.debug("* "+self.myInfo+" -- bb_release",5)
        '''
        self.debug.line(2,"...")
        for job in self.job_list:
            self.debug.debug(job['job'],2)
        self.debug.line(2,"...")
        '''

        temp_bb = 0
        j = 0
        temp_num = len(self.job_list)
        while (j < temp_num):
            if (job_index == self.job_list[j]['job']):
                temp_bb = self.job_list[j]['bb']
                break
            j += 1
        self.idle += temp_bb
        self.avail = self.idle
        self.job_list.pop(j)
        self.debug.debug(
            "  Release" + "[" + str(job_index) + "]" + " Req:" + str(temp_bb) + " Avail:" + str(self.avail) + " ", 4)
        return 1

    def pre_avail(self, proc_num, start, end=None):
        # self.debug.debug("* "+self.myInfo+" -- pre_avail",6)
        # self.debug.debug("pre avail check: "+str(proc_num)+" (" +str(start)+";"+str(end)+")",6)
        if not end or end < start:
            end = start

        i = 0
        temp_job_num = len(self.predict_bb)
        while (i < temp_job_num):
            if (self.predict_bb[i]['time'] >= start and self.predict_bb[i]['time'] < end):
                if (proc_num > self.predict_bb[i]['avail']):
                    return 0
            i += 1
        return 1

    def reserve(self, proc_num, job_index, time, start=None, index=-1):
        # self.debug.debug("* "+self.myInfo+" -- reserve",5)

        temp_max = len(self.predict_bb)
        if (start):
            if (self.pre_avail(proc_num, start, start + time) == 0):
                return -1
        else:
            i = 0
            j = 0
            if (index >= 0 and index < temp_max):
                i = index
            elif (index >= temp_max):
                return -1

            while (i < temp_max):
                if (proc_num <= self.predict_bb[i]['avail']):
                    j = self.find_res_place(proc_num, i, time)
                    if (j == -1):
                        start = self.predict_bb[i]['time']
                        break
                    else:
                        i = j + 1
                else:
                    i += 1

        end = start + time
        j = i

        is_done = 0
        start_index = j
        while (j < temp_max):
            if (self.predict_bb[j]['time'] < end):
                self.predict_bb[j]['idle'] -= proc_num
                self.predict_bb[j]['avail'] = self.predict_bb[j]['idle']
                j += 1
            elif (self.predict_bb[j]['time'] == end):
                is_done = 1
                break
            else:
                self.predict_bb.insert(j, {'time': end, \
                                             'idle': self.predict_bb[j - 1]['idle'],
                                             'avail': self.predict_bb[j - 1]['avail']})
                # self.debug.debug("xx   "+str(proc_num),4)
                self.predict_bb[j]['idle'] += proc_num
                self.predict_bb[j]['avail'] = self.predict_bb[j]['idle']
                is_done = 1

                # self.debug.debug("xx   "+str(n)+"   "+str(k),4)
                break

        if (is_done != 1):
            self.predict_bb.append({'time': end, 'idle': self.tot, 'avail': self.tot})

        self.predict_job.append({'job': job_index, 'start': start, 'end': end})
        '''
        i = 0
        self.debug.line(2,'.')
        temp_num = len(self.predict_bb)
        self.debug.debug("<> "+str(job_index) +"   "+str(proc_num) +"   "+str(time) +"   ",2)
        while (i<temp_num):
            self.debug.debug("O "+str(self.predict_bb[i]),2)
            i += 1
        self.debug.line(2,'.')
        '''
        return start_index

    def pre_delete(self, proc_num, job_index):
        # self.debug.debug("* "+self.myInfo+" -- pre_delete",5)
        return 1

    def pre_modify(self, proc_num, start, end, job_index):
        # self.debug.debug("* "+self.myInfo+" -- pre_modify",5)  
        return 1

    def pre_get_last(self):
        # self.debug.debug("* "+self.myInfo+" -- pre_get_last",6)
        pre_info_last = {'start': -1, 'end': -1}
        for temp_job in self.predict_job:
            # self.debug.debug("xxx   "+str(temp_job),4)
            if (temp_job['start'] > pre_info_last['start']):
                pre_info_last['start'] = temp_job['start']
            if (temp_job['end'] > pre_info_last['end']):
                pre_info_last['end'] = temp_job['end']
        return pre_info_last

    def pre_reset(self, time):
        # self.debug.debug("* "+self.myInfo+" -- pre_reset",5)  
        self.predict_bb = []
        self.predict_job = []
        self.predict_bb.append({'time': time, 'idle': self.idle, 'avail': self.avail})

        temp_job_num = len(self.job_list)
        '''
        i = 0
        self.debug.line(2,'==')
        while (i<temp_job_num):
            self.debug.debug("[] "+str(self.job_list[i]),2)
            i += 1
        self.debug.line(2,'==')
        '''

        i = 0
        j = 0
        while i < temp_job_num:
            if (self.predict_bb[j]['time'] != self.job_list[i]['end'] or i == 0):
                self.predict_bb.append({'time': self.job_list[i]['end'], \
                                          'idle': self.predict_bb[j]['idle'], 'avail': self.predict_bb[j]['avail']})
                j += 1
            self.predict_bb[j]['idle'] += self.job_list[i]['bb']
            self.predict_bb[j]['avail'] = self.predict_bb[j]['idle']
            i += 1
        ''' 
        i = 0
        self.debug.line(2,'..')
        temp_num = len(self.predict_bb)
        while (i<temp_num):
            self.debug.debug("O "+str(self.predict_bb[i]),2)
            i += 1
        self.debug.line(2,'..')
        '''
        return 1

    def find_res_place(self, proc_num, index, time):
        # self.debug.debug("* "+self.myInfo+" -- find_res_place",5)  
        if index >= len(self.predict_bb):
            index = len(self.predict_bb) - 1

        i = index
        end = self.predict_bb[index]['time'] + time
        temp_bb_num = len(self.predict_bb)

        while (i < temp_bb_num):
            if (self.predict_bb[i]['time'] < end):
                if (proc_num > self.predict_bb[i]['avail']):
                    # print "xxxxx   ",temp_bb_num,proc_num,self.predict_bb[i]
                    return i
            i += 1
        return -1