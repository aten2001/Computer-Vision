   	  			  	 		  		  		    	 		 		   		 		  
import numpy as np  	
from scipy import stats	   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
class BagLearner(object):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, learner, kwargs, bags, boost=False,verbose = False):
        self.verbose = verbose
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost

        learner_list = []
        for l in range(bags):
            learner_list.append(learner(**kwargs))
        self.learners = learner_list
        if verbose == True:
            self.get_learner_summary()
        		  	 		  		  		    	 		 		   		 		   			  	 		  		  		    	 		 		   		 		  
    def author(self):  		   	  			  	 		  		  		    	 		 		   		 		  
        return 'shollister7' # replace tb34 with your Georgia Tech username
          	 		  		  		    	 		 		   		 		  
    def addEvidence(self,data,Y):  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Add training data to learner  		   	  			  	 		  		  		    	 		 		   		 		  
        @param data: X values of data to add  		   	  			  	 		  		  		    	 		 		   		 		  
        @param Y: the Y training values  		   	  			  	 		  		  		    	 		 		   		 		  
        """
        for l in self.learners:
            rand_data_slice = np.random.choice(data.shape[0], data.shape[0])
            bag_x = data[rand_data_slice]
            bag_y = Y[rand_data_slice]
            l.addEvidence(bag_x, bag_y)

    
    def query(self,points):  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Estimate a set of test points given the model we built.  		   	  			  	 		  		  		    	 		 		   		 		  
        @param points: should be a numpy array with each row corresponding to a specific query.  		   	  			  	 		  		  		    	 		 		   		 		  
        @returns the estimated values according to the saved model.		   	  			  	 		  		  		    	 		 		   		 		  
        """
        n=points.shape[0]
        result = np.array([0]*n)[np.newaxis]
        output = np.array([])
        for i in range(0, self.bags):
	        new_result = self.learners[i].query(points)
	        new_result = new_result[np.newaxis]
	        result = np.vstack((result , new_result))
            
        result=result[1:,:]
	
        for j in range(0, result.shape[1]):
            m = stats.mode(result[:,j])
            output = np.append( output, m[0][0])
	
        return output
    
    def get_learner_summary(self):
        return "This is the Bag Learner"
        	  			  	 		  		  		    	 		 		   		 		  
        	   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    print("Bag Learner")  		   	  			  	 		  		  		    	 		 		   		 		  
