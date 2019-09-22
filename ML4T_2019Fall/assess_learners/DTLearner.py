   	  			  	 		  		  		    	 		 		   		 		  
import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
class DTLearner(object):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size=1, verbose = False):  		   	  			  	 		  		  		    	 		 		   		 		  
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = np.array([])	   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def author(self):  		   	  			  	 		  		  		    	 		 		   		 		  
        return 'shollister7' # replace tb34 with your Georgia Tech username

    def buildTree(self, data, Y):
        """
        @summary Builds DT by splitting best feature, which is xi
        that has highest absolute correlation with Y.
        @param np array of features (X data)
        @param np array of values 
        @returns np array which is tree [leaf, data.Y, leftchild(None), rightChild(None)]
        

        leaf = np.array([-1, np.mean(Y), np.nan, np.nan])
        
        if data.shape[0] == 1: return leaf
        if len(np.unique(Y)) == 1: return leaf
        if data.shape[0] <= self.leaf_size: return leaf
        
        #find best i to split on, using coeffecients
        possible_feats = range(data.shape[1])
        coeffecients = []
        for index in possible_feats:
            coMatrix = np.corrcoef(data[:,index], Y)
            coeff = coMatrix[0,1]
            coeffecients.append((coeff,index))
        
        coeffecients.sort(reverse=True)
        i= coeffecients[0][1]
        splitVal = np.median(data[:,i])
        
        small = data[:, i] <= splitVal

        if ((np.all(small)) or np.all(~small)):
            return leaf

        lefttree = self.buildTree(data[data[:,i]<=splitVal] , Y[data[:,i]<=splitVal])
        righttree= self.buildTree(data[data[:,i]>splitVal], Y[data[:,i]>splitVal])
        root = np.array([i, splitVal, 1, lefttree.shape[0] + 1])
        append1 = np.append(root, lefttree, 0)
        result = np.append(append1, righttree, 0)
        return result"""
        samples = data.shape[0] #Rows
        features = data.shape[1] #Columns
        #Leaf format is leaf, dataY, left child, right child
        leaf = np.array([-1, np.mean(Y), np.nan, np.nan])
        if (samples <= self.leaf_size): return leaf
        if (len(np.unique(Y)) == 1): return leaf #Check if all values are equal
        if ((np.all(Y==Y[0])) | np.all(data==data[0,:])): return leaf
        split_features = range(features) #List?
        abs_corr = np.abs(np.corrcoef(data, y=Y, rowvar=False))[:-1,-1]
        max_corr = np.nanargmax(abs_corr)
        split_val = np.median(data[:,max_corr])
        smaller_data = data[:,max_corr] <= split_val
        
        if ((np.all(smaller_data)) or np.all(~smaller_data)):
            return leaf
        
        lefttree = self.buildTree(data[smaller_data,:], Y[smaller_data])
        righttree = self.buildTree(data[~smaller_data,:], Y[~smaller_data])
        
        if lefttree.ndim == 1:
            righttree_start = 2
        elif lefttree.ndim > 1:
            righttree_start = lefttree.shape[0] + 1
        root = np.array([max_corr, split_val, 1, righttree_start])

        return np.vstack((root, lefttree, righttree))
        
			  	 		  		  		    	 		 		   		 		  
    def addEvidence(self,dataX,dataY):  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Add training data to learner  		   	  			  	 		  		  		    	 		 		   		 		  
        @param dataX: X values of data to add  		   	  			  	 		  		  		    	 		 		   		 		  
        @param dataY: the Y training values  		   	  			  	 		  		  		    	 		 		   		 		  
        """ 
        self.tree = self.buildTree(dataX, dataY)   	  			  	 		  		  		    	 		 		   		 		  

    def search(self, point, row):
        feat , splitVal = self.tree[row, 0:2]
        if feat == -1:
            return splitVal
        elif point[int(feat)] <= splitVal:
            result = self.search(point, row + int(self.tree[row,2]))
        else:
            result = self.search(point, row+int(self.tree[row,3]))
        return result
    
    def query(self,points):  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Estimate a set of test points given the model we built.  		   	  			  	 		  		  		    	 		 		   		 		  
        @param points: should be a numpy array with each row corresponding to a specific query.  		   	  			  	 		  		  		    	 		 		   		 		  
        @returns the estimated values according to the saved model.		   	  			  	 		  		  		    	 		 		   		 		  
        """
        results = []
        for p in points:
            results.append(self.search(p, 0))
        return np.asarray(results)
    
    def learner_summary(self):
        pass	   	  			  	 		  		  		    	 		 		   		 		  
        	   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    print("DT Learner")  		   	  			  	 		  		  		    	 		 		   		 		  
