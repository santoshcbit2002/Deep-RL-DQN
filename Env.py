# Import routines
from gym import spaces
import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger
lambda_loc_0=2
lambda_loc_1=12
lambda_loc_2=4
lambda_loc_3=7
lambda_loc_4=8

#constants


#Time_matrix = np.load("TM.npy")

class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(i,j) for i in range(1,m+1) for j in range(1,m+1) if i!=j]+[(0,0)]
        self.state_space = [(i,j,k) for i in range(1,m+1) for j in range(1,t+1) for k in range(1,d+1)]
        self.loc_space=np.random.choice(np.arange(1,m+1)) 
        self.hour_of_day=np.random.choice(np.arange(0,t)) 
        self.day_of_week=np.random.choice(np.arange(0,d)) 
        self.total_reward=0
        self.total_time_consumed=0
      
        self.state_init = (self.loc_space,self.hour_of_day,self.day_of_week)
        
        #print('************************************************')
        #print('Action Space - ', self.action_space,'\n')
        #print('Initialial location space - ', self.loc_space,'\n')
        #print('Initialial Hour of Day - ', self.hour_of_day,'\n')
        #print('Initialial Day of Week - ', self.day_of_week,'\n')    
        #print('Initialial State  - ', self.state_init,'\n')  
        #print('Size of State Space  - ', len(self.state_space))  
       
        # Start the first round
        self.reset()
 

    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        location_mat=np.zeros((1,m))
        time_mat=np.zeros((1,t))
        day_mat=np.zeros((1,d))

        #print('location Matrix is : ', location_mat)
        #print('Time Matrix is : ', time_mat)
        #print('Day Matrix is : ', day_mat)
        
        location_mat[0][state[0]-1]=1
        time_mat[0][state[1]]=1
        day_mat[0][state[2]]=1

        #print('location Matrix is : ', location_mat)
        #print('Time Matrix is : ', time_mat)
        #print('Day Matrix is : ', day_mat)
      
        state_encod=np.concatenate((location_mat,time_mat,day_mat),axis=1)
        #print(state_encod)
        #print('shape before transpose ', state_encod.shape)
        state_encod=state_encod.T
        #print('shape after transpose ', state_encod.shape)

        return state_encod

    # Use this function if you are using architecture-2 
    #def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
    # #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        
        #print(' - Called Request Function - ','\n')
        #print(' State Space - ',state)   
        
        location = state[0]
        if location == 1:
            requests = np.random.poisson(lambda_loc_0)
        if location == 2:
            requests = np.random.poisson(lambda_loc_1) 
        if location == 3:
            requests = np.random.poisson(lambda_loc_2) 
        if location == 4:
            requests = np.random.poisson(lambda_loc_3) 
        if location == 5:
            requests = np.random.poisson(lambda_loc_4)

        if requests >15:
            requests =15
        #print(' Requests -  ',requests)   
        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        return possible_actions_index,actions   


    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""

        #print(' - Called Reward Function - ')
        #print(' State Space - ',state)   
        #print(' Action Space - ',action,'\n') 

        start_loc=state[0]
        hour_of_day=state[1]
        dayth_of_week=state[2]
        
        pickup_loc=action[0]
        drop_loc=action[1]
        
        time_spent_for_pickup=int(Time_matrix[start_loc-1][pickup_loc-1][hour_of_day][dayth_of_week])
        pickup_hour=hour_of_day+time_spent_for_pickup
        pickup_day=dayth_of_week
        
        if pickup_hour >= 24 :
           pickup_hour=pickup_hour-24
           pickup_day = pickup_day+1
           if pickup_day >= 7:
               pickup_day=pickup_day-7

        time_spent_for_ride=int(Time_matrix[pickup_loc-1][drop_loc-1][pickup_hour][pickup_day])
     
        reward=(R* time_spent_for_ride) - (C* (time_spent_for_pickup+time_spent_for_ride))
        self.total_reward = self.total_reward + reward
        #print('Total reward : ', self.total_reward)
        return reward


    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        #print(' - Called next_state Function - ')
        #print(' State Space - ',state)   
        #print(' Action Space - ',action,'\n') 

        curr_loc=state[0]
        time_hr=state[1]
        time_day=state[2]
        
        pick_location=action[0]
        drop_location=action[1]
        
        if  action==(0,0):
            next_curr_loc=curr_loc
            next_hr_of_day=time_hr+1  
            next_day=time_day      
        else:    
            next_curr_loc=drop_location
            time_to_pickup=int(Time_matrix[curr_loc-1][pick_location-1][time_hr][time_day])
            time_at_ride=time_hr+time_to_pickup
            day_at_ride=time_day
            
            if time_at_ride >=24:
                time_at_ride=time_at_ride-24
                day_at_ride=day_at_ride+1
                if day_at_ride >= 7:
                    day_at_ride=day_at_ride-7

            time_for_ride=int(Time_matrix[pick_location-1][drop_location-1][time_at_ride][day_at_ride])
            next_hr_of_day=time_for_ride + int(time_at_ride)
            next_day=day_at_ride

        if next_hr_of_day >= 24 :
           next_hr_of_day=next_hr_of_day-24
           next_day = next_day+1
           if next_day >= 7:
               next_day=next_day-7

        next_state=(next_curr_loc,next_hr_of_day,next_day)

        return next_state
    
    def check_if_terminal(self, state, action, Time_matrix):
        """Takes state and action as input and checks if terminal state has arrived"""
        #print(' - Check_if_terminal Function - ')
        #print(' State Space - ',state)   
        #print(' Action Space - ',action,'\n') 

        curr_loc=state[0]
        time_hr=state[1]
        time_day=state[2]
        pick_location=action[0]
        drop_location=action[1]
        
        time_spent_on_pickup=int(Time_matrix[curr_loc-1][pick_location-1][time_hr][time_day])
        time_at_ride=time_spent_on_pickup+time_hr
        day_at_ride=time_day
        if time_at_ride >= 24:
            time_at_ride=time_at_ride-24
            day_at_ride=day_at_ride+1
            if day_at_ride >= 7:
                day_at_ride = day_at_ride-7

        time_spent_on_ride=int(Time_matrix[pick_location-1][drop_location-1][time_at_ride][day_at_ride])
        
        self.total_time_consumed=self.total_time_consumed+time_spent_on_pickup+time_spent_on_ride
        #print('Total Time Consumed : ', self.total_time_consumed)
        if self.total_time_consumed >= 720:
            return True
        else:
            return False
    
    def tracking_info(self):
        return self.total_reward, self.total_time_consumed

    def reset(self):
        return self.action_space, self.state_space, self.state_init