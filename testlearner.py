""""""  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
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
"""  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import math  		  	   		 	   			  		 			     			  	 
import sys  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import numpy as np  	
import matplotlib.pyplot as plt	  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import LinRegLearner as lrl  
import DTLearner as dtl	
import RTLearner as rtl	  	   	
import BagLearner as bl
import InsaneLearner as il

  		  	   		 	   			  		 			     			  	
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)

    header = False
    valid_cols = []
    out = []

    with open(sys.argv[1], 'r') as inf:
        for i in inf:
            row = i.strip().split(",")
            if not header:
                try:
                    [float(j) for j in row]
                    header = True
                    valid_cols = [range(len(row))]
                except:
                    headers = row
                    valid_cols = [k for k, l in enumerate(headers) if 'date' not in l.lower()]
                    header = True
                    continue
            filtered_row = [row[m] for m in valid_cols]
            out.append(list(map(float, filtered_row)))
    data = np.array(out)	   		 	   	

	   		 	   			  		 			     			  	 
    # compute how much of the data is training and testing  		  	   		 	   			  		 			     			  	 
    '''train_rows = int(0.6 * data.shape[0])  		  	   		 	   			  		 			     			  	 
    test_rows = data.shape[0] - train_rows  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    # separate out training and testing data  		  	   		 	   			  		 			     			  	 
    train_x = data[:train_rows, 0:-1]  		  	   		 	   			  		 			     			  	 
    train_y = data[:train_rows, -1]  		  	   		 	   			  		 			     			  	 
    test_x = data[train_rows:, 0:-1]  		  	   		 	   			  		 			     			  	 
    test_y = data[train_rows:, -1]  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    print(f"{test_x.shape}")  		  	   		 	   			  		 			     			  	 
    print(f"{test_y.shape}")  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    # create a learner and train it  		  	   		 	   			  		 			     			  	 
    #learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner  		
    learner=dtl.DTLearner()  	   		 	   			  		 			     			  	 
    learner.add_evidence(train_x, train_y)  # train it  		  	   		 	   			  		 			     			  	 
    print(learner.author())  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    # evaluate in sample  		  	   		 	   			  		 			     			  	 
    pred_y = learner.query(train_x)  # get the predictions  		  	   		 	   			  		 			     			  	 
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		 	   			  		 			     			  	 
    print()  		  	   		 	   			  		 			     			  	 
    print("In sample results")  		  	   		 	   			  		 			     			  	 
    print(f"RMSE: {rmse}")  		  	   		 	   			  		 			     			  	 
    c = np.corrcoef(pred_y, y=train_y)  		  	   		 	   			  		 			     			  	 
    print(f"corr: {c[0,1]}")  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    # evaluate out of sample  		  	   		 	   			  		 			     			  	 
    pred_y = learner.query(test_x)  # get the predictions  		  	   		 	   			  		 			     			  	 
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		 	   			  		 			     			  	 
    print()  		  	   		 	   			  		 			     			  	 
    print("Out of sample results")  		  	   		 	   			  		 			     			  	 
    print(f"RMSE: {rmse}")  		  	   		 	   			  		 			     			  	 
    c = np.corrcoef(pred_y, y=test_y)  		  	   		 	   			  		 			     			  	 
    print(f"corr: {c[0,1]}")'''  	

def make_data(data):	
    train_rows = int(0.6 * data.shape[0])  		  	   		 	   			  		 			     			  	 
    test_rows = data.shape[0] - train_rows   

    train_x = data[:train_rows, 0:-1]  		  	   		 	   			  		 			     			  	 
    train_y = data[:train_rows, -1]  		  	   		 	   			  		 			     			  	 
    test_x = data[train_rows:, 0:-1]  		  	   		 	   			  		 			     			  	 
    test_y = data[train_rows:, -1]  
    return train_x,train_y,test_x,test_y	   		 	   			  		 			     			  	 

def eval_model(model,train_x,train_y,test_x,test_y):
    model.add_evidence(train_x, train_y)

    pred_y = model.query(train_x)  		  	  		     			  	 
    in_rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])   	   		 	   			  		 			     			  	  		  	   		 	   			  		 			     			  	 
    in_c = np.corrcoef(pred_y, y=train_y)[0,1]
		  	   		 	   			  		 			     			  	 
    pred_yt = model.query(test_x)	  	   		 	   			  		 			     			  	 
    out_rmse = math.sqrt(((test_y - pred_yt) ** 2).sum() / test_y.shape[0])  		  	   		 	   			  		 			     			  	  		  	   		 	   			  		 			     			  	 
    out_c = np.corrcoef(pred_yt, y=test_y)[0,1]	
    return in_rmse,in_c,out_rmse,out_c


trax,tray,tex,tey=make_data(data)
final_in_rmse=[]
final_out_rmse=[]
x=[]
for i in range(20, 0, -1):
    learner=dtl.DTLearner(leaf_size=i)
    in_rmse,in_c,out_rmse,out_c=eval_model(learner,tex,tey,trax,tray)
    final_in_rmse.append(in_rmse)
    final_out_rmse.append(out_rmse)
    x.append(i)

plt.plot(final_in_rmse, label='Training Data RMSE')
plt.plot(final_out_rmse, label = 'Test Data RMSE')
plt.title('Experiment 1')
plt.xlabel('DTLearner Leaf Size')
plt.ylabel('RMSE')
plt.ylim(.000,.01)
plt.xticks(range(len(x)), x)
plt.legend()
plt.savefig('./images/Figure_1.png')
plt.close()

final_in_rmse=[]
final_out_rmse=[]
x=[]
for i in range(1,21):
    learner=bl.BagLearner(learner=dtl.DTLearner,kwargs={"leaf_size": 1},bags=i)
    in_rmse,in_c,out_rmse,out_c=eval_model(learner,tex,tey,trax,tray)
    final_in_rmse.append(in_rmse)
    final_out_rmse.append(out_rmse)
    x.append(i)

plt.plot(final_in_rmse, label='Training Data RMSE')
plt.plot(final_out_rmse, label = 'Test Data RMSE')
plt.title('Experiment 2')
plt.xlabel('BagLearner Bag Size')
plt.ylabel('RMSE')
plt.ylim(.000,.01)
plt.xticks(range(len(x)), x)
plt.legend()
plt.savefig('./images/Figure_2.png')
plt.close()


final_in_rmse=[]
final_out_rmse=[]
x=[]
for i in range(20, 0, -1):
    learner=bl.BagLearner(learner=dtl.DTLearner,kwargs={"leaf_size": i},bags=11)
    in_rmse,in_c,out_rmse,out_c=eval_model(learner,tex,tey,trax,tray)
    final_in_rmse.append(in_rmse)
    final_out_rmse.append(out_rmse)
    x.append(i)

plt.plot(final_in_rmse, label='Training Data RMSE')
plt.plot(final_out_rmse, label = 'Test Data RMSE')
plt.title('Experiment 2')
plt.xlabel('BagLearner Leaf Size')
plt.ylabel('RMSE')
plt.ylim(.000,.01)
plt.xticks(range(len(x)), x)
plt.legend()
plt.savefig('./images/Figure_3.png')
plt.close()


in_mae_DT=[]
out_mae_DT=[]
in_mae_RT=[]
out_mae_RT=[]
x=[]
for i in range(20, 0, -1):
    #DT
    learner=dtl.DTLearner(leaf_size=i)
    learner.add_evidence(trax, tray)
    pred_y = learner.query(trax)  		  	  		     			  	 
    in_MAE= np.abs(tray - pred_y).sum() / tray.shape[0]
    in_mae_DT.append(in_MAE)

    learner=dtl.DTLearner(leaf_size=i)
    learner.add_evidence(trax, tray)
    pred_y = learner.query(tex)  		  	  		     			  	 
    out_MAE= np.abs(tey - pred_y).sum() / tey.shape[0]
    out_mae_DT.append(out_MAE)

    #RT
    learner=rtl.RTLearner(leaf_size=i)
    learner.add_evidence(trax, tray)
    pred_y = learner.query(trax)  		  	  		     			  	 
    in_MAE= np.abs(tray - pred_y).sum() / tray.shape[0]
    in_mae_RT.append(in_MAE)

    learner=rtl.RTLearner(leaf_size=i)
    learner.add_evidence(trax, tray)
    pred_y = learner.query(tex)  		  	  		     			  	 
    out_MAE= np.abs(tey - pred_y).sum() / tey.shape[0]
    out_mae_RT.append(out_MAE)
    x.append(i)

plt.plot(in_mae_DT, label='DT Training Data MAE')
plt.plot(out_mae_DT, label='DT Test Data MAE')
plt.plot(in_mae_RT, label = 'RT Training Data MAE')
plt.plot(out_mae_RT, label = 'RT Test Data MAE')
plt.title('Experiment 3')
plt.xlabel('DT/RTLearner Leaf Size')
plt.ylabel('MAE')
plt.xticks(range(len(x)), x)
plt.legend()
plt.savefig('./images/Figure_4.png')
plt.close()


in_R2_DT=[]
out_R2_DT=[]
in_R2_RT=[]
out_R2_RT=[]
x=[]
for i in range(20, 0, -1):
    #DT
    learner=dtl.DTLearner(leaf_size=i)
    learner.add_evidence(trax, tray)
    pred_y = learner.query(trax)  		  	  		     			  	 
    in_R2= 1 - np.sum((tray - pred_y)**2) / np.sum((tray - np.mean(tray)) ** 2)
    in_R2_DT.append(in_R2)

    learner=dtl.DTLearner(leaf_size=i)
    learner.add_evidence(trax, tray)
    pred_y = learner.query(tex)  		  	  		     			  	 
    out_R2= 1 - np.sum((tey - pred_y)**2) / np.sum((tey - np.mean(tey)) ** 2)
    out_R2_DT.append(out_R2)

    #RT
    learner=rtl.RTLearner(leaf_size=i)
    learner.add_evidence(trax, tray)
    pred_y = learner.query(trax)  		  	  		     			  	 
    in_R2= 1 - np.sum((tray - pred_y)**2) / np.sum((tray - np.mean(tray)) ** 2)
    in_R2_RT.append(in_R2)

    learner=rtl.RTLearner(leaf_size=i)
    learner.add_evidence(trax, tray)
    pred_y = learner.query(tex)  		  	  		     			  	 
    out_R2= 1 - np.sum((tey - pred_y)**2) / np.sum((tey - np.mean(tey)) ** 2)
    out_R2_RT.append(out_R2)
    x.append(i)

plt.plot(in_R2_DT, label='DT Training Data R^2')
plt.plot(out_R2_DT, label='DT Test Data R^2')
plt.plot(in_R2_RT, label = 'RT Training Data R^2')
plt.plot(out_R2_RT, label = 'RT Test Data R^2')
plt.title('Experiment 3')
plt.xlabel('DT/RTLearner Leaf Size')
plt.ylabel('R^2')
plt.xticks(range(len(x)), x)
plt.legend()
plt.savefig('./images/Figure_5.png')
plt.close()