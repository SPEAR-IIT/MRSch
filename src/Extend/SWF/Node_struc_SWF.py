from datetime import datetime
import time
import re
#import CqSim.bb_struc as Class_bb_struc
import CqSim.Node_struc as Class_Node_struc
import sys

__metaclass__ = type

class Node_struc_SWF(Class_Node_struc.Node_struc):


    def node_allocate(self, proc_num, bb, job_index, start, end):
        #self.debug.debug("* "+self.myInfo+" -- node_allocate",5)
        if self.is_available(proc_num) == 0 or self.bb_is_available(bb)==0:
            return 0

        i = 0
        for node in self.nodeStruc:
            if node['state'] < 0:
                node['state'] = job_index
                node['start'] = start
                node['end'] = end
                i += 1
            # self.debug.debug("  yyy: "+str(node['state'])+"   "+str(job_index),4)
            if (i >= proc_num):
                break
        i=0
        for bbfer in self.bbStruc:
            if bbfer['state'] < 0:
                bbfer['state'] = job_index
                bbfer['start'] = start
                bbfer['end'] = end
                i += 1
            # self.debug.debug("  yyy: "+str(node['state'])+"   "+str(job_index),4)
            if (i >= bb):
                break

        self.idle -= proc_num
        self.avail = self.idle
        self.bbidle-=bb
        self.bbavail=self.bbidle
        temp_job_info = {'job':job_index, 'end': end, 'node': proc_num,'bb':bb}
        j = 0
        is_done = 0
        temp_num = len(self.job_list)
        while (j<temp_num):
            if (temp_job_info['end']<self.job_list[j]['end']):
                self.job_list.insert(j,temp_job_info)
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
        self.debug.debug("  Allocate"+"["+str(job_index)+"]"+" Req:"+str(proc_num)+"Node Avail:"+str(self.avail)+"BB Avail:"+str(self.bbavail) +" ",4)
        return 1
        
    def node_release(self, job_index, end):
        #self.debug.debug("* "+self.myInfo+" -- node_release",5)
        '''
        self.debug.line(2,"...")
        for job in self.job_list:
            self.debug.debug(job['job'],2)
        self.debug.line(2,"...")
        '''

        for node in self.nodeStruc:
            # self.debug.debug("  xxx: "+str(node['state'])+"   "+str(job_index),4)
            if node['state'] == job_index:
                node['state'] = -1
                node['start'] = -1
                node['end'] = -1



        for bbfer in self.bbStruc:
            # self.debug.debug("  xxx: "+str(node['state'])+"   "+str(job_index),4)
            if bbfer['state'] == job_index:
                bbfer['state'] = -1
                bbfer['start'] = -1
                bbfer['end'] = -1


        temp_node = 0
        temp_bb=0
        j = 0
        temp_num = len(self.job_list)

        while (j<temp_num):
            if (job_index==self.job_list[j]['job']):
                temp_node = self.job_list[j]['node']
                temp_bb=self.job_list[j]['bb']
                break
            j += 1

        self.idle += temp_node
        self.avail = self.idle
        self.bbidle += temp_bb
        self.bbavail = self.bbidle
        self.job_list.pop(j)
        self.debug.debug("  Release"+"["+str(job_index)+"]"+" Req:"+str(temp_node)+"Node Avail:"+str(self.avail)+"BB Avail:"+str(self.bbavail) +" ",4)
        return 1
        
    def pre_avail(self, proc_num, bb, start, end = None):
        #self.debug.debug("* "+self.myInfo+" -- pre_avail",6)
        #self.debug.debug("pre avail check: "+str(proc_num)+" (" +str(start)+";"+str(end)+")",6)
        if not end or end < start:
            end = start

        i = 0
        temp_job_num = len(self.predict_node)
        while (i < temp_job_num):
            if (self.predict_node[i]['time']>=start and self.predict_node[i]['time']<end):
                if ((proc_num>self.predict_node[i]['avail'])or (bb>self.predict_node[i]['bbavail'])):
                    return 0
            i += 1
        return 1
        
    def reserve(self, proc_num, bb,job_index, time, start = None, index = -1 ):
        #self.debug.debug("* "+self.myInfo+" -- reserve",5)
            
        temp_max = len(self.predict_node)
        if (start):
            if (self.pre_avail(proc_num,bb,start,start+time)==0):
                return -1
        else:
            i = 0
            j = 0
            if (index >= 0 and index < temp_max):
                i = index
            elif(index >= temp_max):
                return -1
            
            while (i<temp_max): 
                if (proc_num<=self.predict_node[i]['avail'] and bb<=self.predict_node[i]['bbavail']):

                    j = self.find_res_place(proc_num,bb,i,time)

                    if (j == -1):
                        start = self.predict_node[i]['time']
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
            if (self.predict_node[j]['time']<end):
                self.predict_node[j]['idle'] -= proc_num
                self.predict_node[j]['avail'] = self.predict_node[j]['idle']
                self.predict_node[j]['bbidle'] -= bb
                self.predict_node[j]['bbavail'] = self.predict_node[j]['bbidle']
                j += 1
            elif (self.predict_node[j]['time']==end):
                is_done = 1
                break
            else:
                self.predict_node.insert(j,{'time':end,\
                 'idle':self.predict_node[j-1]['idle'], 'avail':self.predict_node[j-1]['avail'],'bbidle':self.predict_node[j-1]['bbidle'], 'bbavail':self.predict_node[j-1]['bbavail']})
                #self.debug.debug("xx   "+str(proc_num),4)
                self.predict_node[j]['idle'] += proc_num
                self.predict_node[j]['avail'] = self.predict_node[j]['idle']
                self.predict_node[j]['bbidle'] += bb
                self.predict_node[j]['bbavail'] = self.predict_node[j]['bbidle']
                is_done = 1
                
                #self.debug.debug("xx   "+str(n)+"   "+str(k),4)
                break
            
        if (is_done != 1):
            self.predict_node.append({'time':end,'idle':self.tot,'avail':self.tot,'bbidle':self.bbtot,'bbavail':self.bbtot})
                
        self.predict_job.append({'job':job_index, 'start':start, 'end':end})
        '''
        i = 0
        self.debug.line(2,'.')
        temp_num = len(self.predict_node)
        self.debug.debug("<> "+str(job_index) +"   "+str(proc_num) +"   "+str(time) +"   ",2)
        while (i<temp_num):
            self.debug.debug("O "+str(self.predict_node[i]),2)
            i += 1
        self.debug.line(2,'.')
        ''' 
        return start_index
     
    def pre_delete(self, proc_num, job_index):
        #self.debug.debug("* "+self.myInfo+" -- pre_delete",5)
        return 1
        
    def pre_modify(self, proc_num, start, end, job_index):  
        #self.debug.debug("* "+self.myInfo+" -- pre_modify",5)  
        return 1
        
    def pre_get_last(self):
        #self.debug.debug("* "+self.myInfo+" -- pre_get_last",6)
        pre_info_last= {'start':-1, 'end':-1}
        for temp_job in self.predict_job:
            #self.debug.debug("xxx   "+str(temp_job),4)
            if (temp_job['start']>pre_info_last['start']):
                pre_info_last['start'] = temp_job['start']
            if (temp_job['end']>pre_info_last['end']):
                pre_info_last['end'] = temp_job['end']
        return pre_info_last
        
    def pre_reset(self, time):
        #self.debug.debug("* "+self.myInfo+" -- pre_reset",5)  
        self.predict_node = []
        self.predict_job = []
        self.predict_node.append({'time':time, 'idle':self.idle, 'avail':self.avail,'bbidle':self.bbidle, 'bbavail':self.bbavail})
                            
                            
        temp_job_num = len(self.job_list)

        
        i = 0
        j = 0
        while i< temp_job_num:
            if (self.predict_node[j]['time']!=self.job_list[i]['end'] or i == 0):
                self.predict_node.append({'time':self.job_list[i]['end'],\
                                    'idle':self.predict_node[j]['idle'], 'avail':self.predict_node[j]['avail'],'bbidle':self.predict_node[j]['bbidle'], 'bbavail':self.predict_node[j]['bbavail']})
                j += 1
            self.predict_node[j]['idle'] += self.job_list[i]['node']
            self.predict_node[j]['avail'] = self.predict_node[j]['idle']
            self.predict_node[j]['bbidle'] += self.job_list[i]['bb']
            self.predict_node[j]['bbavail'] = self.predict_node[j]['bbidle']
            i += 1

        return 1
        
    
    def find_res_place(self, proc_num,bb, index, time):
        #self.debug.debug("* "+self.myInfo+" -- find_res_place",5)  
        if index>=len(self.predict_node):
            index = len(self.predict_node) - 1
             
        i = index
        end = self.predict_node[index]['time']+time
        temp_node_num = len(self.predict_node)
        
        while (i < temp_node_num):
            if (self.predict_node[i]['time']<end):
                if (proc_num>self.predict_node[i]['avail'] or bb>self.predict_node[i]['bbavail'] ):
                    #print "xxxxx   ",temp_node_num,proc_num,self.predict_node[i]
                    return i
            i += 1
        return -1