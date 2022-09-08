import IOModule.Log_print as Log_print
import sys
import pickle
import random
from random import choice
import numpy as np
from collections import deque
import time


from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Merge, \
    Activation, Embedding
from keras.optimizers import SGD, Adam, rmsprop
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import tensorflow as tf
__metaclass__ = type

class Cqsim_sim:
    def __init__(self, module, debug = None, monitor = None,is_training='1'):
        self.myInfo = "Cqsim Sim"
        self.module = module
        self.debug = debug
        self.monitor = monitor
        self.is_training = int(is_training)
        self.debug.line(4," ")
        self.debug.line(4,"#")
        self.debug.debug("# "+self.myInfo,1)
        self.debug.line(4,"#")
        
        self.event_seq = []
        #self.event_pointer = 0
        self.monitor_start = 0
        self.current_event = None
        #obsolete
        self.job_num = len(self.module['job'].job_info())
        self.currentTime = 0

        self.timestamp=[]
        self.timeseries=[]
        self.action=[]
        #obsolete
        self.read_job_buf_size = 100
        self.read_job_pointer = 0 # next position in job list
        self.previous_read_job_time = -1 # lastest read job submit time
        self.learning_rate=0.0001
        self.model = None
        self.action_size = 4
        self.nodeutil_time=[]
        self.bbutil_time=[]
        self.batch_size=6
        self.measurement_size=2
        self.timesteps=2
        self.count=0
        self.measurement_input = np.zeros((self.batch_size, self.measurement_size))
        self.goal = np.array([0.5, 0.5] * self.timesteps)
        self.measurement_target = np.zeros((self.batch_size, (self.measurement_size * self.timesteps)))
        self.goal_input= np.tile(self.goal, (self.batch_size, 1))
        self.state_size = 4 * self.action_size + 100 * 2 + 20 * 2
        self.state_seq= np.zeros((self.batch_size, self.state_size))


        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.0001
        self.explore=200


        self.debug.line(4)
        for module_name in self.module:
            temp_name = self.module[module_name].myInfo
            self.debug.debug(temp_name+" ................... Load",4)
            self.debug.line(4)
        
    def reset(self, module = None, debug = None, monitor = None):
        #self.debug.debug("# "+self.myInfo+" -- reset",5)
        if module:
            self.module = module
        
        if debug:
            self.debug = debug
        if monitor:
            self.monitor = monitor
               
            
        self.event_seq = []
        #self.event_pointer = 0
        self.monitor_start = 0
        self.current_event = None
        #obsolete
        self.job_num = len(self.module['job'].job_info())
        self.currentTime = 0
        #obsolete
        self.read_job_buf_size = 100
        self.read_job_pointer = 0
        self.previous_read_job_time = -1

    def cqsim_sim(self):
        state_size = self.state_size
        measurement_size = self.measurement_size
        time_steps = self.timesteps
        goal_size = measurement_size * time_steps
        action_size = self.action_size
        learning_rate = self.learning_rate
        self.model = self.dfp_network(state_size, measurement_size, goal_size, action_size, time_steps, learning_rate)
        if self.is_training != 0:


            self.import_submit_events()
            # self.insert_event_job()

            self.insert_event_extend()

            self.scan_event()

            self.print_result()
            self.model.save_weights("model.h5",overwrite=True)
            self.debug.debug("------ weights saved!", 2)
            self.debug.debug("------ Simulating Done!", 2)


        else:
            self.model.load_weights("model.h5")
            print('................... Loading weights......................')
            self.import_submit_events()
            # self.insert_event_job()

            self.insert_event_extend()
            time1 = time.time()
            self.scan_event()

            self.print_result()
            self.debug.debug("------ Simulating Done!", 2)
            self.debug.debug(lvl=1)



        return


    def dfp_network(self,input_shape, measurement_size, goal_size, action_size, num_timesteps, learning_rate):
        """
        Neural Network for Direct Future Predition (DFP)
        """

        # Perception Feature
        state_input = Input(shape=(input_shape,))
        perception_feat = Dense(4000, activation='relu')(state_input)
        perception_feat = Dense(1000, activation='relu')(perception_feat)
        perception_feat = Dense(512, activation='tanh')(perception_feat)

        # Measurement Feature
        measurement_input = Input(shape=((measurement_size,)))
        measurement_feat = Dense(128, activation='relu')(measurement_input)
        measurement_feat = Dense(128, activation='relu')(measurement_feat)
        measurement_feat = Dense(128, activation='relu')(measurement_feat)

        # Goal Feature
        goal_input = Input(shape=((goal_size,)))
        goal_feat = Dense(128, activation='relu')(goal_input)
        goal_feat = Dense(128, activation='relu')(goal_feat)
        goal_feat = Dense(128, activation='relu')(goal_feat)

        concat_feat = merge([perception_feat, measurement_feat, goal_feat], mode='concat')

        measurement_pred_size = measurement_size * num_timesteps  # 2 measurements, 6 timesteps

        expectation_stream = Dense(measurement_pred_size, activation='tanh')(concat_feat)

        prediction_list = []
        for i in range(action_size):
            action_stream = Dense(measurement_pred_size, activation='tanh')(concat_feat)
            prediction_list.append(merge([action_stream, expectation_stream], mode='sum'))

        model = Model(input=[state_input, measurement_input, goal_input], output=prediction_list)

        adam = Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return model

    def get_action(self, state, measurement, goal, inference_goal,is_training):
        """
        Get action from model using epsilon-greedy policy
        """
        if is_training!=0:
            if np.random.rand() <= self.epsilon:
                # print("----------Random Action----------")
                action_idx = random.randrange(self.action_size)
            else:
                measurement = np.expand_dims(measurement, axis=0)
                goal = np.expand_dims(goal, axis=0)
                state = np.expand_dims(state, axis=0)
                f = self.model.predict([state, measurement, goal])
                f_pred = np.vstack(f)
                obj = np.sum(np.multiply(f_pred, inference_goal), axis=1)

                action_idx = np.argmax(obj)
            if self.epsilon > self.final_epsilon:
                self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore
        else:
            measurement = np.expand_dims(measurement, axis=0)
            goal = np.expand_dims(goal, axis=0)
            state = np.expand_dims(state, axis=0)
            f = self.model.predict([state, measurement, goal])
            f_pred = np.vstack(f)  # 3x6
            obj = np.sum(np.multiply(f_pred, inference_goal), axis=1)

            action_idx = np.argmax(obj)
        return action_idx

    def import_submit_events(self):
        # fread jobs to job list and buffer to event_list dynamically
        if self.read_job_pointer < 0:
            return -1
        temp_return = self.module['job'].dyn_import_job_file()
        i = self.read_job_pointer
        #while (i < len(self.module['job'].job_info())):
        while (i < self.module['job'].job_info_len()):
            self.insert_event(1,self.module['job'].job_info(i)['submit'],2,[1,i])
            self.previous_read_job_time = self.module['job'].job_info(i)['submit']
            self.debug.debug("  "+"Insert job["+"2"+"] "+str(self.module['job'].job_info(i)['submit']),4)
            i += 1
        #print("Insert Jobs!")
        if temp_return == None or temp_return < 0 :
            self.read_job_pointer = -1
            return -1
        else:
            self.read_job_pointer = i
            return 0

    #obsolete
    def insert_submit_events(self):
        # first read all jobs to job list, buffer to event_list dynamically
        #self.debug.debug("# "+self.myInfo+" -- insert_event_job",5) 
        if self.read_job_pointer < 0:
            return -1
        i = self.read_job_pointer
        while (i < self.read_job_buf_size + self.read_job_pointer and i < self.job_num):
            self.insert_event(1,self.module['job'].job_info(i)['submit'],2,[1,i])
            self.previous_read_job_time = self.module['job'].job_info(i)['submit']
            self.debug.debug("  "+"Insert job["+"2"+"] "+str(self.module['job'].job_info(i)['submit']),4)
            i += 1
        if i >= self.job_num:
            self.read_job_pointer = -1
        else:
            self.read_job_pointer = i
        return 0
    
    #obsolete
    def insert_event_job(self):
        #self.debug.debug("# "+self.myInfo+" -- insert_event_job",5) 
        i = 0
        while (i < self.job_num):
            self.insert_event(1,self.module['job'].job_info(i)['submit'],2,[1,i])
            self.debug.debug("  "+"Insert job["+"2"+"] "+str(self.module['job'].job_info(i)['submit']),4)
            i += 1
        return
    
    def insert_event_monitor(self, start, end):
        #self.debug.debug("# "+self.myInfo+" -- insert_event_monitor",5) 
        if (not self.monitor):
            return -1
        temp_num = start/self.monitor
        temp_num = int(temp_num)
        temp_time = temp_num*self.monitor

        #self.monitor_start=self.event_pointer
        self.monitor_start=0

        i = 0
        while (temp_time < end):
            if (temp_time>=start):
                self.insert_event(2,temp_time,5,None)
                self.debug.debug("  "+"Insert mon["+"5"+"] "+str(temp_time),4)
            temp_time += self.monitor
        return
    
    def insert_event_extend(self):
        #self.debug.debug("# "+self.myInfo+" -- insert_event_extend",5) 
        return
    
    def insert_event(self, type, time, priority, para = None):
        #self.debug.debug("# "+self.myInfo+" -- insert_event",5) 
        temp_index = -1
        new_event = {"type":type, "time":time, "prio":priority, "para":para}
        if (type == 1):
            #i = self.event_pointer
            i = 0
            while (i<len(self.event_seq)):
                if (self.event_seq[i]['time']==time):
                    if (self.event_seq[i]['prio']>priority):
                        temp_index = i
                        break
                elif (self.event_seq[i]['time']>time):
                    temp_index = i
                    break 
                i += 1
        elif (type == 2):
            temp_index = self.get_index_monitor()
            
        if (temp_index>=len(self.event_seq) or temp_index == -1):
            self.event_seq.append(new_event)
        else:
            self.event_seq.insert(temp_index,new_event)
            
    
    def delete_event(self, type, time, index):
        #self.debug.debug("# "+self.myInfo+" -- delete_event",5) 
        return
    
    def get_index_monitor (self):
        #self.debug.debug("# "+self.myInfo+" -- get_index_monitor",5) 
        '''
        if (self.event_pointer>=self.monitor_start):
            self.monitor_start=self.event_pointer+1
        temp_mon = self.monitor_start
        self.monitor_start += 1
        return temp_mon
        '''
        self.monitor_start += 1
        return self.monitor_start

    
    def scan_event(self):
       # self.debug.debug("# "+self.myInfo+" -- scan_event",5) 
        self.debug.line(2," ")
        self.debug.line(2,"=")
        self.debug.line(2,"=")
        self.current_event = None
        #while (self.event_pointer < len(self.event_seq) or self.read_job_pointer >= 0):
        while (len(self.event_seq) > 0 or self.read_job_pointer >= 0):
            #print('event_seq',len(self.event_seq))
            if len(self.event_seq) > 0:
                temp_current_event = self.event_seq[0]
                temp_currentTime = temp_current_event['time']
            else:
                temp_current_event = None
                temp_currentTime = -1
            #if (temp_currentTime >= self.previous_read_job_time or self.event_pointer >= len(self.event_seq)) and self.read_job_pointer >= 0:
            if (len(self.event_seq) == 0 or temp_currentTime >= self.previous_read_job_time) and self.read_job_pointer >= 0:
                #print('insert_submit_events from scan_event',temp_currentTime >= self.previous_read_job_time,(self.event_pointer >= len(self.event_seq) and self.read_job_pointer >= 0))
                #self.insert_submit_events()
                self.import_submit_events()
                continue
            self.current_event = temp_current_event
            self.currentTime = temp_currentTime
            if (self.current_event['type'] == 1):
                self.debug.line(2," ") 
                self.debug.line(2,">>>") 
                self.debug.line(2,"--") 
                #print ("  Time: "+str(self.currentTime)) 
                self.debug.debug("  Time: "+str(self.currentTime),2) 
                self.debug.debug("   "+str(self.current_event),2)
                self.debug.line(2,"--") 
                self.debug.debug("  Wait: "+str(self.module['job'].wait_list()),2) 
                self.debug.debug("  Run : "+str(self.module['job'].run_list()),2) 
                self.debug.line(2,"--") 
                self.debug.debug(" Node and BB info-- Node Tot:"+str(self.module['node'].get_tot())+" Node Idle:"+str(self.module['node'].get_idle())+" Avail:"+str(self.module['node'].get_avail())+ " BB total:"+str(self.module['node'].bb_get_tot())+" BB Idle:"+str(self.module['node'].bb_get_idle())+" BB Avail:"+str(self.module['node'].bb_get_avail())+" ",2)
                #self.debug.debug(" BB info-- Tot:" + str(self.module['bb'].get_tot()) + " Idle:" + str(self.module['bb'].get_idle()) + " Avail:" + str(self.module['bb'].get_avail()) + " ", 2)
                self.debug.line(2,"--") 
                
                self.event_job(self.current_event['para'])
            elif (self.current_event['type'] == 2):
                self.event_monitor(self.current_event['para'])
            elif (self.current_event['type'] == 3):
                self.event_extend(self.current_event['para'])
            self.sys_collect()
            self.interface()
            #self.event_pointer += 1
            del self.event_seq[0]
        self.debug.line(2,"=")
        self.debug.line(2,"=")
        self.debug.line(2," ")
        return
    
    def event_job(self, para_in = None):
        #self.debug.debug("# "+self.myInfo+" -- event_job",5) 
        '''
        self.debug.line(2,"xxxxx")
        i = 0
        while (i<len(self.event_seq)):
            self.debug.debug(self.event_seq[i],2) 
            i += 1
            
        self.debug.line(2,"xxxxx")
        self.debug.line(2," ")
        self.debug.line(2," ")
        '''
        if (self.current_event['para'][0] == 1):
            self.submit(self.current_event['para'][1])
        elif (self.current_event['para'][0] == 2):
            self.finish(self.current_event['para'][1])
       #self.score_calculate()
        self.start_scan()
        #if (self.event_pointer < len(self.event_seq)-1):
        if (len(self.event_seq) > 1):
            #self.insert_event_monitor(self.currentTime, self.event_seq[self.event_pointer+1]['time'])
            self.insert_event_monitor(self.currentTime, self.event_seq[1]['time'])
        return
    
    def event_monitor(self, para_in = None):
        #self.debug.debug("# "+self.myInfo+" -- event_monitor",5) 
        self.alg_adapt()
        self.window_adapt()
        self.print_adapt(None)
        return
    
    def event_extend(self, para_in = None):
        #self.debug.debug("# "+self.myInfo+" -- event_extend",5) 
        return
    
    def submit(self, job_index):
        #self.debug.debug("# "+self.myInfo+" -- submit",5) 
        self.debug.debug("[Submit]  "+str(job_index),3)
        self.module['job'].job_submit(job_index)
        return
    
    def finish(self, job_index):
        #self.debug.debug("# "+self.myInfo+" -- finish",5) 
        self.debug.debug("[Finish]  "+str(job_index),3)
        self.module['node'].node_release(job_index,self.currentTime)
        self.module['job'].job_finish(job_index)
        self.module['output'].print_result(self.module['job'], job_index)
        self.module['job'].remove_job_from_dict(job_index)
        if self.is_training!=0:
            time1 = self.currentTime
            tot = self.module['node'].get_tot()
            idle = self.module['node'].get_idle()
            nodeuti = float(tot - idle) / tot
            bbtot = self.module['node'].bb_get_tot()
            bbidle = self.module['node'].bb_get_idle()
            bbuti = float(bbtot - bbidle) / bbtot
            if (len(self.timeseries)!=0) and (time1==self.timeseries[-1]):

                self.nodeutil_time[-1]=nodeuti
                self.bbutil_time[-1]=bbuti

            else:
                self.timeseries.append(time1)
                self.nodeutil_time.append(nodeuti)
                self.bbutil_time.append(bbuti)



        return
    
    def start(self, job_index):
        #self.debug.debug("# "+self.myInfo+" -- start",5) 
        self.debug.debug("[Start]  "+str(job_index),3)
        self.module['node'].node_allocate(self.module['job'].job_info(job_index)['reqProc'],self.module['job'].job_info(job_index)['reqMem'], job_index,\
         self.currentTime, self.currentTime + self.module['job'].job_info(job_index)['reqTime'])

        self.module['job'].job_start(job_index, self.currentTime)
        self.insert_event(1,self.currentTime+self.module['job'].job_info(job_index)['run'],1,[2,job_index])
        if self.is_training!=0:
            time1 = self.currentTime
            tot = self.module['node'].get_tot()
            idle = self.module['node'].get_idle()
            nodeuti = float(tot - idle) / tot
            bbtot = self.module['node'].bb_get_tot()
            bbidle = self.module['node'].bb_get_idle()
            bbuti = float(bbtot - bbidle) / bbtot
            if (len(self.timeseries) != 0) and (time1 == self.timeseries[-1]):
                self.nodeutil_time[-1] = nodeuti
                self.bbutil_time[-1] = bbuti

            else:
                self.timeseries.append(time1)
                self.nodeutil_time.append(nodeuti)
                self.bbutil_time.append(bbuti)

        return

    def start_scan(self):
        # self.debug.debug("# "+self.myInfo+" -- start_scan",5)
        if self.is_training!=0:
            while True:
                temp_nodeStruc = self.module['node'].get_nodeStruc()
                temp_bbStruc=self.module['node'].get_bbStruc()
                temp_wait = self.module['job'].wait_list()
                wait_num = len(temp_wait)
                curr=self.currentTime

                if wait_num == 0:
                    break

                index=self.count
                state,cpuhour,bbhour=self.make_feature_vector(temp_wait,temp_nodeStruc,temp_bbStruc,curr)
                self.state_seq[index,:]=state
                tot = self.module['node'].get_tot()
                idle = self.module['node'].get_idle()
                node_util = float(tot - idle) / tot

                bbtot = self.module['node'].bb_get_tot()
                bbidle = self.module['node'].bb_get_idle()
                bb_util = float(bbtot - bbidle) / bbtot
                if (len(self.timeseries) == 0) or (curr != self.timeseries[-1]):
                    self.timeseries.append(curr)
                    self.nodeutil_time.append(node_util)
                    self.bbutil_time.append(bb_util)
                measure=np.array([node_util,bb_util])
                self.measurement_input[index,:]=measure
                cpu_weight=cpuhour/(cpuhour+bbhour)
                bb_weight=1-cpu_weight
                goal = np.array([cpu_weight,bb_weight] * self.timesteps)
                self.goal_input[index,:]=goal
                inference_goal=goal # can use different for training and test[0,1]
                max_id = self.get_action(state, measure, goal, inference_goal,self.is_training)
                self.action.append(max_id)

                if (max_id<len(temp_wait)) :
                    temp_wait= [temp_wait[max_id]] + temp_wait[:max_id] + temp_wait[max_id+1:]
                    self.timestamp.append(self.currentTime)
                else:
                    self.timestamp.append(-1)
                temp_job = self.module['job'].job_info(temp_wait[0])
                if (self.module['node'].is_available(temp_job['reqProc']) and self.module['node'].bb_is_available(
                        temp_job['reqMem'])):
                    self.start(temp_wait[0])
                else:
                    temp_wait = self.module['job'].wait_list()
                    self.backfill(temp_wait)
                    break
                self.count+=1
                if self.count>=self.batch_size:

                    if len(self.action)==0:
                        return
                    measure_predict=self.model.predict([self.state_seq, self.measurement_input, self.goal_input])
                    for i in range(self.batch_size):
                        idx=self.action[i]
                        time_stamp=self.timestamp[i]
                        if time_stamp==-1:
                            measure_predict[idx][i, :] =np.zeros(self.measurement_size*self.timesteps)
                        else:
                            temp_list=[]
                            time_index=self.timeseries.index(time_stamp)
                            k=0
                            while(k<self.timesteps):
                                if time_index>=len(self.timeseries):
                                    cpu_util = 0.0
                                    bb_util = 0.0
                                else:
                                    cpu_util=self.nodeutil_time[time_index]
                                    bb_util=self.bbutil_time[time_index]
                                temp_list.append(cpu_util)
                                temp_list.append(bb_util)
                                k+=1
                                time_index+=1
                            measure_predict[idx][i, :] = np.array(temp_list)
                    self.model.train_on_batch([self.state_seq, self.measurement_input, self.goal_input], measure_predict)

                    self.refresh()
        else:
            while True:
                temp_nodeStruc = self.module['node'].get_nodeStruc()
                temp_bbStruc = self.module['node'].get_bbStruc()
                # start_max = self.module['win'].start_num()
                temp_wait = self.module['job'].wait_list()

                wait_num = len(temp_wait)
                curr = self.currentTime
                if wait_num == 0:
                    break

                state, cpuhour, bbhour = self.make_feature_vector(temp_wait, temp_nodeStruc, temp_bbStruc, curr)
                tot = self.module['node'].get_tot()
                idle = self.module['node'].get_idle()
                node_util = float(tot - idle) / tot
                bbtot = self.module['node'].bb_get_tot()
                bbidle = self.module['node'].bb_get_idle()
                bb_util = float(bbtot - bbidle) / bbtot
                measure = np.array([node_util, bb_util])
                # set goal_vector
                cpu_weight = cpuhour / (cpuhour + bbhour)
                bb_weight = 1 - cpu_weight
                goal = np.array([cpu_weight, bb_weight] * self.timesteps)
                inference_goal = goal  # you can use different for training and test, such as ransom[0,1] for training
                max_id = self.get_action(state, measure, goal, inference_goal,self.is_training)


                if (max_id < len(temp_wait)):
                    temp_wait = [temp_wait[max_id]] + temp_wait[:max_id] + temp_wait[max_id + 1:]
                temp_job = self.module['job'].job_info(temp_wait[0])
                if (self.module['node'].is_available(temp_job['reqProc']) and self.module['node'].bb_is_available(
                        temp_job['reqMem'])):
                    self.start(temp_wait[0])
                else:
                    temp_wait = self.module['job'].wait_list()
                    self.backfill(temp_wait)
                    break

            # self.debug.debug("# "+self.myInfo+" -- start_scan",5)

        return





    def refresh(self):
        self.timestamp = []
        self.timeseries = []
        self.action=[]
        self.bbutil_time=[]
        self.nodeutil_time=[]
        self.count=0
        self.measurement_input = np.zeros((self.batch_size, self.measurement_size))
        self.goal = np.array([0.5, 0.5] * self.timesteps)
        self.goal_input = np.tile(self.goal, (self.batch_size, 1))
        self.state_seq = np.zeros((self.batch_size, self.state_size))







    def start_window(self, temp_wait_B):
        #self.debug.debug("# "+self.myInfo+" -- start_window",5) 
        win_size = self.module['win'].window_size()
        
        if (len(temp_wait_B)>win_size):
            temp_wait_A = temp_wait_B[0:win_size]
            temp_wait_B = temp_wait_B[win_size:]
        else:
            temp_wait_A = temp_wait_B
            temp_wait_B = []

        temp_wait_info = []
        max_num = len(temp_wait_A)
        i = 0
        while (i < max_num):
            temp_job = self.module['job'].job_info(temp_wait_A[i])
            temp_wait_info.append({"index":temp_wait_A[i],"proc":temp_job['reqProc'],"bb":temp_job['reqMem'],\
             "node":temp_job['reqProc'],"run":temp_job['run'],"score":temp_job['score']})
            i += 1 
            
        temp_wait_A = self.module['win'].start_window(temp_wait_info,{"time":self.currentTime})
        temp_wait_B[0:0] = temp_wait_A
        return temp_wait_B
    
    def backfill(self, temp_wait):
        #self.debug.debug("# "+self.myInfo+" -- backfill",5) 
        temp_wait_info = []
        max_num = len(temp_wait)
        i = 0
        while (i < max_num):
            temp_job = self.module['job'].job_info(temp_wait[i])
            temp_wait_info.append({"index":temp_wait[i],"proc":temp_job['reqProc'],"bb":temp_job['reqMem'],\
             "node":temp_job['reqProc'],"run":temp_job['run'],"score":temp_job['score']})
            i += 1
        backfill_list = self.module['backfill'].backfill(temp_wait_info, {'time':self.currentTime})
        #self.debug.debug("HHHHHHHHHHHHH "+str(backfill_list)+" -- backfill",2) 
        if not backfill_list:
            return 0
        
        for job in backfill_list:
            self.start(job)
        return 1
    
    def sys_collect(self):
        #self.debug.debug("# "+self.myInfo+" -- sys_collect",5) 
        '''
        temp_inter = 0
        if (self.event_pointer+1<len(self.event_seq)):
            temp_inter = self.event_seq[self.event_pointer+1]['time'] - self.currentTime
        temp_size = 0
        
        event_code=None
        if (self.event_seq[self.event_pointer]['type'] == 1):
            if (self.event_seq[self.event_pointer]['para'][0] == 1):   
                event_code='S'
            elif(self.event_seq[self.event_pointer]['para'][0] == 2):   
                event_code='E'
        elif (self.event_seq[self.event_pointer]['type'] == 2):
            event_code='Q'
        '''
        temp_inter = 0
        if (len(self.event_seq) > 1):
            temp_inter = self.event_seq[1]['time'] - self.currentTime
        temp_size = 0
        
        event_code=None
        if (self.event_seq[0]['type'] == 1):
            if (self.event_seq[0]['para'][0] == 1):   
                event_code='S'
            elif(self.event_seq[0]['para'][0] == 2):   
                event_code='E'
        elif (self.event_seq[0]['type'] == 2):
            event_code='Q'
        temp_info = self.module['info'].info_collect(time=self.currentTime, event=event_code,\
         uti=(self.module['node'].get_tot()-self.module['node'].get_idle())*1.0/self.module['node'].get_tot(),\
         waitNum=len(self.module['job'].wait_list()),waitSize=self.module['job'].node_wait_size_acc(),  inter=temp_inter)
        self.print_sys_info(temp_info)
        return
    
    def interface(self, sys_info = None):
        #self.debug.debug("# "+self.myInfo+" -- interface",5) 
        return
    
    def alg_adapt(self):
        #self.debug.debug("# "+self.myInfo+" -- alg_adapt",5) 
        return 0
    
    def window_adapt(self):
        #self.debug.debug("# "+self.myInfo+" -- window_adapt",5) 
        return 0
    
    def print_sys_info(self, sys_info):
        #self.debug.debug("# "+self.myInfo+" -- print_sys_info",5) 
        self.module['output'].print_sys_info(sys_info)
    
    def print_adapt(self, adapt_info):
        #self.debug.debug("# "+self.myInfo+" -- print_adapt",5) 
        self.module['output'].print_adapt(adapt_info)
    
    def print_result(self):
        #self.debug.debug("# "+self.myInfo+" -- print_result",5) 
        self.module['output'].print_sys_info()
        self.debug.debug(lvl=1)
        self.module['output'].print_result(self.module['job'])



    def make_feature_vector(self, wait_job, node_struc, bb_struc,currTime):
        accumu_bb=0
        accumu_cpu=0
        vector = []
        waitNum = len(wait_job)
        temp_wait=[]
        i = 0
        while (i < waitNum):
            temp_job = self.module['job'].job_info(wait_job[i])
            temp_wait.append(temp_job)
            i += 1
        i = 0


        while (i < self.action_size):
            if (i<waitNum):
                s = float(temp_wait[i]['submit'])
                t = float(temp_wait[i]['reqTime'])
                n = float(temp_wait[i]['reqProc'])
                w = float(currTime - s)
                b= float(temp_wait[i]['reqMem'])

                info = [w, t, n, b]
                vector=vector+info
                accumu_cpu+=n*t
                accumu_bb+=b*t
            else:
                info=[0,0,0,0]
                vector=vector+info
            i += 1


        for node in node_struc:
            info = []
            if node['state'] < 0:
                info.append(1)
                info.append(0)
            else:
                info.append(0)
                rem=node['end'] - currTime
                info.append(rem)
                accumu_cpu+=rem
            vector=vector+info

        for bb in bb_struc:
            info = []
            if bb['state'] < 0:
                info.append(1)
                info.append(0)
            else:
                info.append(0)
                rem = bb['end'] - currTime
                info.append(rem)
                accumu_bb += rem

            vector = vector + info
        vector=np.array(vector)
        return vector,accumu_cpu,accumu_bb



