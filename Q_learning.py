#!  user/bin/python

import numpy as np
import pandas as pd
import time

N_STATES = 6 # Number of the states 
ACTIONS = ['left','right'] # the actions for each state
EPSILON = 0.9 # The greedy rate for randomly choosing the actions
ALPHA = 0.1 # The learning rate for the Q learning
GAMMA = 0.9 # The decay rate for the future q value
MAX_EPISODES = 15 # The maximum episode
FRESH_TIME = 0.05 # The interval for moving between the non-terminal states

np.random.seed(1)


# Create the Q table 
def build_q_table(n_states,actions):
	table = pd.DataFrame(np.zeros((n_states,len(actions))),columns=actions,)
	return table

# The action choosen function
def choose_action(s,q_table):
	state_actions = q_table.iloc[s,:] # choose all the actions for the given state
	if (np.random.uniform()>EPSILON) or (state_actions.all() == 0): # the random mode or the initialization mode 
		action_name = np.random.choice(ACTIONS)
	else:
		action_name = state_actions.argmax() # the greedy mode 
	return action_name


# Define the feedback function from the environment
def get_env_feedback(s,a):
	"This is how agent will interact with the environmet"
	if a == 'right':
		if s == N_STATES - 2:
			s_ = 'terminal'
			r  = 1
		else:
			s_ = s + 1
			r = 0
	else: # turn to left
		r = 0
		if s == 0:
			s_ = s
		else:
			s_ = s - 1
	return s_,r


# Define the env update function
def update_env(s,episode,step_counter):
	"This is how environmen be updated"
	env_list = ['-']*(N_STATES-1)+['T'] # '------T' our environment
	if s == 'terminal':
		interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
		print('\r{}'.format(interaction), end='')
		time.sleep(2)
		print('\r                               ',end='')
	else:
		env_list[s] = 'o'
		interaction = ''.join(env_list)
		print('\r{}'.format(interaction),end='')
		time.sleep(FRESH_TIME)
'''
def update_env(s, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if s == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[s] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)
'''

 # The main loop for the RL
def rl():
	q_table = build_q_table(N_STATES,ACTIONS) # Create the q table for this system
	for episode in range(MAX_EPISODES):
		step_counter = 0
		s = 0 # The initialization state for every episode
		is_terminated = False # Indicate whether it's the end of this episode
		update_env(s,episode,step_counter) # Update the env with the initialized state
		while not is_terminated:
			a = choose_action(s,q_table)
			s_,r = get_env_feedback(s,a) # get the feedback from the env with this action under the state
			q_predict = q_table.ix[s,a] # estimate the q(s,a) value
			if s_ != 'terminal':
				q_target = r + GAMMA * q_table.iloc[s_,:].max() # The actual q(s,a) value, this episode has not terminated
			else:
				q_target = r # the actual q(s,a), episode has terminated
				is_terminated = True # terminate this episode
			q_table.ix[s,a] += ALPHA * (q_target - q_predict) # q_table update
			s = s_ # Move to the next state
			step_counter += 1
			update_env(s,episode,step_counter) # Env update
			#step_counter +=1
	return q_table				

if __name__ == "__main__":
	q_table = rl()
	print ('\r\nQ-table:\n')
	print(q_table)
		
