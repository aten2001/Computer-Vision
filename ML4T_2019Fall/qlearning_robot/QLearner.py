"""  		   	  			  	 		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		   	  			  	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		   	  			  	 		  		  		    	 		 		   		 		  
All Rights Reserved  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		   	  			  	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		   	  			  	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			  	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			  	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			  	 		  		  		    	 		 		   		 		  
or edited.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		   	  			  	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		   	  			  	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			  	 		  		  		    	 		 		   		 		  
GT honor code violation.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		   	  			  	 		  		  		    	 		 		   		 		  
GT User ID: shollister7 (replace with your User ID)  		   	  			  	 		  		  		    	 		 		   		 		  
GT ID: 903304661 (replace with your GT ID)  		   	  			  	 		  		  		    	 		 		   		 		  
"""  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
import random as rand  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
class QLearner(object):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
        self.verbose = verbose  		   	  			  	 		  		  		    	 		 		   		 		  
        self.num_actions = num_actions  		   	  			  	 		  		  		    	 		 		   		 		  
        self.s = 0  		   	  			  	 		  		  		    	 		 		   		 		  
        self.a = 0  

        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.num_states = num_states
        self.gamma = gamma
        self.alpha = alpha
        self.q = np.zeros((self.num_states, self.num_actions))
        self.setup_table(self.dyna) 	  			  	 		  		  		    	 		 		   		 		  

    
    def querysetstate(self, s):  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Update the state without updating the Q-table  		   	  			  	 		  		  		    	 		 		   		 		  
        @param s: The new state  		   	  			  	 		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        self.s = s  		   	  			  	 		  		  		    	 		 		   		 		  
        action = rand.randint(0, self.num_actions-1) 
        if rand.random() > self.rar:
            action = np.argmax(self.q[s,])
        #self.rar = self.rar * self.radr
        #self.a = action	   	  			  	 		  		  		    	 		 		   		 		  
        if self.verbose: print(f"s = {s}, a = {action}")  		   	  			  	 		  		  		    	 		 		   		 		  
        return action  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def query(self,s_prime,r):  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Update the Q table and return an action  		   	  			  	 		  		  		    	 		 		   		 		  
        @param s_prime: The new state  		   	  			  	 		  		  		    	 		 		   		 		  
        @param r: The ne state  		   	  			  	 		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			  	 		  		  		    	 		 		   		 		  
        """ 
        #self.setup_table(self.dyna)	   	  			  	 		  		  		    	 		 		   		 		  
        #action = rand.randint(0, self.num_actions-1)  		   	  			  	 		  		  		    	 		 		   		 		  
         		   	  			  	 		  		  		    	 		 		   		 		  
        self.q[self.s, self.a] = (1 - self.alpha) * self.q[self.s, self.a] + self.alpha * (r + self.gamma * np.max(self.q[s_prime, :]))

        if self.rar >= rand.random():
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.q[s_prime,:])
        
        self.rar = self.rar * self.radr

        if self.dyna:
            # increment Tc, update T and R
            self.Tc[self.s, self.a, s_prime] = self.Tc[self.s, self.a, s_prime] + 1
            self.T = self.Tc / self.Tc.sum(axis=2, keepdims=True)
            self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + (self.alpha * r)
            #print(self.R)

            # iterate through the dyna simulations
            for i in range(0, self.dyna):
                # select a random a and s
                a_dyna = np.random.randint(low=0, high=self.num_actions)
                s_dyna = np.random.randint(low=0, high=self.num_states)
                # infer s' from T
                s_prime_dyna = np.random.multinomial(1, self.T[s_dyna, a_dyna,]).argmax()
                #print("S prime Dyna : {}".format(s_prime_dyna))
                #s_prime_test = self.T[s_dyna, a_dyna]
                #print("S_prime_test : {}".format(s_prime_test))
                # compute R from s and a
                r = self.R[s_dyna, a_dyna]
                # update q
                
                #print("r {}".format(r))
                #print("a_dyna {}".format(a_dyna))
                #print("s_dyna {}".format(s_dyna))
                #print("s_prime_dyna {}".format(s_prime_dyna))
                self.q[s_dyna, a_dyna] = (1 - self.alpha) * self.q[s_dyna, a_dyna] + \
                                         self.alpha * (r + self.gamma * np.max(self.q[s_prime_dyna,:]))
                            

        self.s = s_prime
        self.a = action

        if self.verbose: print(f"s = {s_prime}, a = {action}, r={r}") 
        
        return action  		   	  			  	 		  		  		    	 		 		   		 		  
    
    def setup_table(self, dyna):
        table_size = (self.num_states, self.num_actions)
        self.q = np.random.uniform(-1, 1, table_size)
        
        if dyna:
            self.Tc = np.full((self.num_states,self.num_actions,self.num_states),0.0001)
            self.T = self.Tc / self.Tc.sum(axis=2, keepdims=True)
            self.R = np.full(table_size,-1.0)
            
    def author(self):
        return "shollister7"

if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    print("Remember Q from Star Trek? Well, this isn't him")  		   	  			  	 		  		  		    	 		 		   		 		  
