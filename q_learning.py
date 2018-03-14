import gridWorld
import numpy as np
import random

def q_learn(start_state, episodes, gamma, epsilon):
	# initialize Q and N
	Q = np.zeros((17, 4))
	N = np.zeros((17, 4)) # N(s,a): num times a executed from state s
	
	for i in range(episodes):
		#print i
		s = start_state
		
		while True:
			# stop when the terminal state is reached
			if (s == 16):
				break
				
			a = select_action(Q, epsilon, s)
			N[s, a] += 1
		
			next_states_probab = T[s, :, a]
			action_outcomes = np.sort(next_states_probab[next_states_probab > 0])
		
			# simulate transition model: get s' after executing a from state s
			rand = random.random()
			if len(action_outcomes) == 3: 
				if (rand < 0.9):
					s_prime = np.where(next_states_probab == 0.9)[0][0]
				elif (rand < 0.95):
					s_prime = np.where(next_states_probab == 0.05)[0][0]
				else:
					s_prime = np.where(next_states_probab == 0.05)[0][1]		
			elif len(action_outcomes) == 2: 
				if (rand < 0.95):
					s_prime = np.where(next_states_probab == max(next_states_probab))[0][0]
				else:
					s_prime = np.where(next_states_probab == 0.05)[0][0]
			elif len(action_outcomes) == 1:
				s_prime = np.where(next_states_probab == 1)[0][0]
		
# 			print 'state', s
# 			print 'action', a
# 			print 'next state', s_prime
		
			alpha = 1 / N[s, a]
			Q[s, a] = Q[s, a] + alpha * (R[s] + gamma * max(Q[s_prime, :]) - Q[s, a])
# 			print Q
# 			print '\n\n'
			s = s_prime

	return Q	
	
def select_action(Q, epsilon, s):
	rand = random.random()
	actions = [0, 1, 2, 3]
	if rand < epsilon:
		action = random.choice(actions)
	else:
		action = np.argmax(Q[s])
	return action
	
def get_utilities(Q):
	U = -np.ones(17) # optimal policy (tells us which action to take from each state)
	
	for s in range(17):
		U[s] = max(Q[s])
	return U	
	
def get_policies(Q):
	policies = -np.ones(17) # optimal policy (tells us which action to take from each state)
	
	for s in range(17):
		policies[s] = np.argmax(Q[s])
	return policies
		

def main():

	a = 0.9
	b = 0.05
	gamma = 0.99
	
	episodes = 10000 # each episode consists of a sequence of moves from the start state 
	# until the end state is reached.
	
	epsilon = 0.2 # probability of selecting a random action

	# for q-learning problem, transition and reward models are used only to simulate the
	# environment when an agent executes an action:
	global T, R
	T = gridWorld.gridWorld(a, b)[0]
	R = gridWorld.gridWorld(a, b)[1]
	
	# 0='UP', 1='DOWN', 2='LEFT', 3='ROIGHT'	

	start_state = 4
	Q = q_learn(start_state, episodes, gamma, epsilon)
	#print Q

	U = get_utilities(Q)	
	policies = get_policies(Q)
	
	print '\nUtilities:\n', U[:-1].reshape(4,4)
	print '\nPolicies (0=''UP'', 1=''DOWN'', 2=''LEFT'', 3=''RIGHT''):\n', \
			policies[:-1].reshape(4,4)
			
# 	print '\nUtilities:\n', U
# 	print '\nPolicies:\n', policies
	
if __name__ == "__main__":
    main()

