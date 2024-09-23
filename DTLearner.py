		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import numpy as np  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
class DTLearner(object):  		  	   		 	   			  		 			     			  	 
  	   		 	   			  		 			     			  	 
    def __init__(self, verbose=False,leaf_size=1):  	
        self.leaf_size=leaf_size	  	
        self.verbose=verbose   	
        self.tree=None	 	   			  		 			     			  	 	  	   		 	   			  		 			     			  	 
        pass		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    def author(self):  		  	   		 	   			  		 			     			  	   	   		 	   			  		 			     			  	 
        return "efigone3"
    
    def build_dtree(self, data_x, data_y):
        if data_x.shape[0] <= self.leaf_size or np.all(data_y == data_y[0]):
            return np.array([[-1, np.median(data_y), np.nan, np.nan]])

        feature = 0
        corr_max = 0
        for i in range(data_x.shape[1]):
            if np.std(data_x[:, i]) == 0 or np.std(data_y) == 0:
                corr = 0 
            else:
                corr = np.corrcoef(data_x[:, i], data_y)[0, 1]
            if abs(corr) > abs(corr_max):
                feature = i
                corr_max = corr

        split_val = np.median(data_x[:, feature])

        if not np.any(data_x[:, feature] <= split_val) or not np.any(data_x[:, feature] > split_val):
            return np.array([[-1, np.median(data_y), np.nan, np.nan]])

        left_tree = self.build_dtree(data_x[data_x[:, feature] <= split_val], data_y[data_x[:, feature] <= split_val])
        right_tree = self.build_dtree(data_x[data_x[:, feature] > split_val], data_y[data_x[:, feature] > split_val])

        root = np.array([[feature, split_val, 1, left_tree.shape[0] + 1]])
        return np.vstack((root, left_tree, right_tree))

    def add_evidence(self, data_x, data_y):  		  	   		 	   			  		 			     			  	 	  	   		 	   			  		 			     			  	 
        self.tree=self.build_dtree(data_x,data_y)
    
    def query(self, points):
        pred = np.array([])
        for i in points:
            c_pos = 0
            while c_pos < self.tree.shape[0]:
                t_pos = self.tree[c_pos]
                feature = int(t_pos[0])
                split = t_pos[1]
                if int(t_pos[0]) == -1:
                    pred = np.append(pred, t_pos[1])
                    break
                if i[feature] <= split:
                    c_pos += 1
                else:
                    c_pos += int(t_pos[3])
        return pred	  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
