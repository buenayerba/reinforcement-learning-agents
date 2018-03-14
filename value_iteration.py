import gridWorld
import numpy as np

def valueIteration(trans_model, rewards, gamma, epsilon):
	UPrime = np.zeros(17) # U and UPrime are vectors of utilitues
		
	i = 0
	
	while True:
		i += 1 
		U = np.copy(UPrime)
		delta = 0 # max change in utility of any state in an iteration
		best_actions = -np.ones(17) # optimal policy (tells us which action to take from each state)

		for s in range(17):		
			max_act_util = float('-inf')
			best_act_s = -1
			for action in range(4):
	
				#print '\n\n\n\nstate', s, 'action', action
				#print trans_model[s, :, action]
	
				act_util_vector = np.multiply(trans_model[s, :, action], U)
				act_util = np.sum(act_util_vector)
				
				if act_util > max_act_util:
					max_act_util = act_util
					best_act_s = action
			
			# Update UPrime and corresponding best_actions		
			UPrime[s] = rewards[s] + gamma * max_act_util
			best_actions[s] = best_act_s
			
			if abs(UPrime[s] - U[s]) > delta:
				delta = UPrime[s] - U[s]

		if delta <= epsilon:
			break
			
	#print 'i:', i # number of iterations
	return U, best_actions
	

def main():
	# transition parameters
	a = 0.9  # intended move
	b = 0.05  # lateral move
	
	discount = 0.99
	eps = 0.01
	
	T = gridWorld.gridWorld(a, b)[0]
	R = gridWorld.gridWorld(a, b)[1]
	
	# 0='UP', 1='DOWN', 2='LEFT', 3='ROIGHT'	
	
	U, best_actions = valueIteration(T, R, discount, eps)
	print '\nUtilities:\n', U[:16].reshape(4,4)
	
	
	print '\nPolicies (0=''UP'', 1=''DOWN'', 2=''LEFT'', 3=''RIGHT''):\n', \
			best_actions[:16].reshape(4,4)
	print '\nUtilities:\n', U
	print '\nPolicies\n', best_actions
	
if __name__ == "__main__":
    main()

