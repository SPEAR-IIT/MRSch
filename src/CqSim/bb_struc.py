from datetime import datetime
import time
import re

__metaclass__ = type


class bb_struc:
    def __init__(self, debug=None):
        self.myInfo = "bb Structure"
        self.debug = debug
        self.bbStruc = []
      #  self.job_list = []
      #  self.predict_bb = []
      #  self.predict_job = []
        self.bbtot = -1
        self.bbidle = -1
        self.bbavail = -1

        #self.debug.line(4, " ")
        #self.debug.line(4, "#")
       # self.debug.debug("# " + self.myInfo, 1)
        #self.debug.line(4, "#")

    def reset(self, debug=None):
        # self.debug.debug("* "+self.myInfo+" -- reset",5)
       # self.debug = debug
        self.bbStruc = []
     #   self.job_list = []
      #  self.predict_bb = []
        self.bbtot = -1
        self.bbidle = -1
        self.bbavail = -1

    def read_list(self, source_str):
        # self.debug.debug("* "+self.myInfo+" -- read_list",5)
        result_list = []
        regex_str = "[\[,]([^,\[\]]*)"
        result_list = re.findall(regex_str, source_str)
        for item in result_list:
            item = int(item)
        return result_list

    def import_bb_file(self, bb_file):
        # self.debug.debug("* "+self.myInfo+" -- import_bb_file",5)
        regex_str = "([^;\\n]*)[;\\n]"
        bbFile = open(bb_file, 'r')
        self.bbStruc = []

        i = 0
        while (1):
            tempStr = bbFile.readline()
            if not tempStr:  # break when no more line
                break
            temp_dataList = re.findall(regex_str, tempStr)

           # self.debug.debug("  bb[" + str(i) + "]: " + str(temp_dataList), 4)
            tempInfo = {"id": int(temp_dataList[0]), \
                        "location": self.read_list(temp_dataList[1]), \
                        "group": int(temp_dataList[2]), \
                        "state": int(temp_dataList[3]), \
                        "bb": int(temp_dataList[4]), \
                        "start": -1, \
                        "end": -1, \
                        "extend": None}
            self.bbStruc.append(tempInfo)
            i += 1
        bbFile.close()
        self.bbtot = len(self.bbStruc)
        self.bbidle = self.bbtot
        self.bbavail = self.bbtot
        print ("***********")
        print ("bbtotal",self.bbtot)
        print ("bbidle",self.bbidle)
        print ("bbavail",self.bbavail)
        #self.debug.debug("  Tot:" + str(self.tot) + " bbidle:" + str(self.bbidle) + " bbavail:" + str(self.bbavail) + " ", 4)
        return

    def import_bb_config(self, config_file):
        # self.debug.debug("* "+self.myInfo+" -- import_bb_config",5)
        regex_str = "([^=\\n]*)[=\\n]"
        bbFile = open(config_file, 'r')
        config_data = {}

        #self.debug.line(4)
        while (1):
            tempStr = bbFile.readline()
            if not tempStr:  # break when no more line
                break
            temp_dataList = re.findall(regex_str, tempStr)
            config_data[temp_dataList[0]] = temp_dataList[1]
            #self.debug.debug(str(temp_dataList[0]) + ": " + str(temp_dataList[1]), 4)
        #self.debug.line(4)
        bbFile.close()

    def import_bb_data(self, bb_data):
        # self.debug.debug("* "+self.myInfo+" -- import_bb_data",5)
        self.bbStruc = []

        temp_len = len(bb_data)
        i = 0
        while (i < temp_len):
            temp_dataList = bb_data[i]

            tempInfo = {"id": temp_dataList[0], \
                        "location": temp_dataList[1], \
                        "group": temp_dataList[2], \
                        "state": temp_dataList[3], \
                        "proc": temp_dataList[4], \
                        "start": -1, \
                        "end": -1, \
                        "extend": None}
            self.bbStruc.append(tempInfo)
            i += 1
        self.bbtot = len(self.bbStruc)
        self.bbidle = self.bbtot
        self.bbavail = self.bbtot

    def bb_is_available(self, proc_num):
        # self.debug.debug("* "+self.myInfo+" -- is_available",6)
        result = 0
        print ("******************************Here we go**************")
        if self.bbavail >= proc_num:
            result = 1
        #self.debug.debug("[bbavail Check] " + str(result), 6)
        return result

    def bb_get_tot(self):
        # self.debug.debug("* "+self.myInfo+" -- get_tot",6)
        return self.bbtot

    def bb_get_idle(self):
        # self.debug.debug("* "+self.myInfo+" -- get_idle",6)
        return self.bbidle

    def bb_get_avail(self):
        # self.debug.debug("* "+self.myInfo+" -- get_avail",6)
        return self.bbavail
'''
    def bb_allocate(self, proc_num, job_index, start, end):
        # self.debug.debug("* "+self.myInfo+" -- bb_allocate",5)
        if self.is_available(proc_num) == 0:
            return 0
        i = 0
        for bb in self.bbStruc:
            if bb['state'] < 0:
                bb['state'] = job_index
                bb['start'] = start
                bb['end'] = end
                i += 1
            # self.debug.debug("  yyy: "+str(bb['state'])+"   "+str(job_index),4)
            if (i >= proc_num):
                break
        self.bbidle -= proc_num
  
        return 1

    def bb_release(self, job_index, end):
        # self.debug.debug("* "+self.myInfo+" -- bb_release",5)
        i = 0
        for bb in self.bbStruc:
            # self.debug.debug("  xxx: "+str(bb['state'])+"   "+str(job_index),4)
            if bb['state'] == job_index:
                bb['state'] = -1
                bb['start'] = -1
                bb['end'] = -1
                i += 1
        if i <= 0:
            self.debug.debug("  Release Fail!", 4)
            return 0
        self.bbidle += i
        self.bbavail = self.bbidle
        j = 0
        temp_num = len(self.job_list)
        while (j < temp_num):
            if (job_index == self.job_list[j]['job']):
                break
            j += 1
        self.job_list.pop(j)
        self.debug.debug(
            "  Release" + "[" + str(job_index) + "]" + " Req:" + str(i) + " bbavail:" + str(self.bbavail) + " ", 4)
        return 1

    def bb_pre_avail(self, proc_num, start, end=None):
        # self.debug.debug("* "+self.myInfo+" -- pre_avail",6)
        # self.debug.debug("pre bbavail check: "+str(proc_num)+" (" +str(start)+";"+str(end)+")",6)
        if not end or end < start:
            end = start

        i = 0
        temp_job_num = len(self.predict_bb)
        while (i < temp_job_num):
            if (self.predict_bb[i]['time'] >= start and self.predict_bb[i]['time'] < end):
                if (proc_num > self.predict_bb[i]['bbavail']):
                    return 0
            i += 1
        return 1
    
    def bb_reserve(self, proc_num, job_index, time, start=None, index=-1):
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
                if (proc_num <= self.predict_bb[i]['bbavail']):
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
                k = 0
                n = 0
                while k < self.tot and n < proc_num:
                    if (self.predict_bb[j]['bb'][k] == -1):
                        self.predict_bb[j]['bb'][k] = job_index
                        self.predict_bb[j]['bbidle'] -= 1
                        self.predict_bb[j]['bbavail'] = self.predict_bb[j]['bbidle']
                        n += 1
                    k += 1
                j += 1
            elif (self.predict_bb[j]['time'] == end):
                is_done = 1
                break
            else:
                temp_list = []
                k = 0
                while k < self.tot:
                    temp_list.append(self.predict_bb[j - 1]['bb'][k])
                    k += 1
                self.predict_bb.insert(j, {'time': end, 'bb': temp_list, \
                                             'bbidle': self.predict_bb[j - 1]['bbidle'],
                                             'bbavail': self.predict_bb[j - 1]['bbavail']})
                k = 0
                n = 0
                # self.debug.debug("xx   "+str(proc_num),4)
                while k < self.tot and n < proc_num:
                    if (self.predict_bb[j]['bb'][k] == job_index):
                        self.predict_bb[j]['bb'][k] = -1
                        self.predict_bb[j]['bbidle'] += 1
                        self.predict_bb[j]['bbavail'] = self.predict_bb[j]['bbidle']
                        n += 1
                    k += 1
                is_done = 1

                # self.debug.debug("xx   "+str(n)+"   "+str(k),4)
                break

        temp_list = []
        if (is_done != 1):
            k = 0
            while k < self.tot:
                temp_list.append(-1)
                k += 1
            self.predict_bb.append({'time': end, 'bb': temp_list, \
                                      'bbidle': self.tot, 'bbavail': self.tot})

        self.predict_job.append({'job': job_index, 'start': start, 'end': end})
     
        return start_index

    def bb_pre_delete(self, proc_num, job_index):
        # self.debug.debug("* "+self.myInfo+" -- pre_delete",5)
        return 1

    def bb_pre_modify(self, proc_num, start, end, job_index):
        # self.debug.debug("* "+self.myInfo+" -- pre_modify",5)  
        return 1

    def bb_pre_get_last(self):
        # self.debug.debug("* "+self.myInfo+" -- pre_get_last",6)
        pre_info_last = {'start': -1, 'end': -1}
        for temp_job in self.predict_job:
            # self.debug.debug("xxx   "+str(temp_job),4)
            if (temp_job['start'] > pre_info_last['start']):
                pre_info_last['start'] = temp_job['start']
            if (temp_job['end'] > pre_info_last['end']):
                pre_info_last['end'] = temp_job['end']
        return pre_info_last

    def bb_pre_reset(self, time):
        # self.debug.debug("* "+self.myInfo+" -- pre_reset",5)  
        self.predict_bb = []
        self.predict_job = []
        temp_list = []
        i = 0
        while i < self.tot:
            temp_list.append(self.bbStruc[i]['state'])
            i += 1
        self.predict_bb.append({'time': time, 'bb': temp_list, \
                                  'bbidle': self.bbidle, 'bbavail': self.bbavail})

        temp_job_num = len(self.job_list)
        i = 0
        j = 0
        while i < temp_job_num:
            if (self.predict_bb[j]['time'] != self.job_list[i]['end'] or i == 0):
                temp_list = []
                k = 0
                while k < self.tot:
                    temp_list.append(self.predict_bb[j]['bb'][k])
                    k += 1
                self.predict_bb.append({'time': self.job_list[i]['end'], 'bb': temp_list, \
                                          'bbidle': self.predict_bb[j]['bbidle'], 'bbavail': self.predict_bb[j]['bbavail']})
                j += 1
            k = 0
            while k < self.tot:
                if (self.predict_bb[j]['bb'][k] == self.job_list[i]['job']):
                    self.predict_bb[j]['bb'][k] = -1
                    self.predict_bb[j]['bbidle'] += 1
                k += 1
            i += 1
            self.predict_bb[j]['bbavail'] = self.predict_bb[j]['bbidle']
      
        return 1

    def bb_find_res_place(self, proc_num, index, time):
        self.debug.debug("* " + self.myInfo + " -- find_res_place", 5)
        if index >= len(self.predict_bb):
            index = len(self.predict_bb) - 1

        i = index
        end = self.predict_bb[index]['time'] + time
        temp_bb_num = len(self.predict_bb)

        while (i < temp_bb_num):
            if (self.predict_bb[i]['time'] < end):
                if (proc_num > self.predict_bb[i]['bbavail']):
                    # print "xxxxx   ",temp_bb_num,proc_num,self.predict_bb[i]
                    return i
            i += 1
        return -1
'''