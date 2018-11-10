import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 
import scipy 
import math  
import sklearn
import random
from Function_Defs_Perceptron import * 
#Initialization of Bias
Bias  = 0.1
Learning_Rate = 0.1

Weight_Vector = np.zeros((25,1))
for i in range(len(Weight_Vector)):
	Weight_Vector[i]  = round(random.uniform(-0.5,0.5), 2)
	random.seed(1)

### 1 PATTERNS ###

#TRAINING DATA 1 
Ones_Training_Pattern_1 = correct_pattern_generator()

#TRAINING DATA 2
Ones_Training_Pattern_2 = correct_pattern_generator()
Ones_Training_Pattern_2[1]=1
Ones_Training_Pattern_2[21:23]=1

#TRAINING DATA 3
Ones_Training_Pattern_3 = correct_pattern_generator()
Ones_Training_Pattern_3[6]=1

#TRAINING DATA 4
Ones_Training_Pattern_4 = np.zeros((25,1))
Ones_Training_Pattern_4[3] = 1
Ones_Training_Pattern_4[7:9] = 1
Ones_Training_Pattern_4[12] = 1
Ones_Training_Pattern_4[16:18] = 1
Ones_Training_Pattern_4[21] = 1

#TRAINING DATA 5 
Ones_Training_Pattern_5 = correct_pattern_generator()
Ones_Training_Pattern_5[2]=0
Ones_Training_Pattern_5[4]=1


#TRAINING DATA 6
Ones_Training_Pattern_6 = np.zeros((25,1))
Ones_Training_Pattern_6[20:4:-4]=1
Ones_Training_Pattern_6[4]=1



###END OF 1 PATTERNS###


### 0 PATTERNS ###


#Training Pattern Zeros 1
correct_pattern_ZEROS_1= np.zeros((25,1))
correct_pattern_ZEROS_1.shape = (5,5)
correct_pattern_ZEROS_1[0, 1:3] = 1
correct_pattern_ZEROS_1[0:4, 1] = 1
correct_pattern_ZEROS_1[0:4, 3] = 1
correct_pattern_ZEROS_1[4, 1:3] = 1
correct_pattern_ZEROS_1.shape= (25,1)

#Training Pattern Zeros 2
correct_pattern_ZEROS_2= np.zeros((25,1))
correct_pattern_ZEROS_2.shape = (5,5)
correct_pattern_ZEROS_2[0, 2] = 1
correct_pattern_ZEROS_2[1:4, 1] = 1
correct_pattern_ZEROS_2[1:4, 3] = 1
correct_pattern_ZEROS_2[4, 2] = 1
correct_pattern_ZEROS_2.shape= (25,1)


#Training Pattern Zeros 3
correct_pattern_ZEROS_3= np.zeros((25,1))
correct_pattern_ZEROS_3.shape = (5,5)
correct_pattern_ZEROS_3[0, 2] = 1
correct_pattern_ZEROS_3[1:5, 1] = 1
correct_pattern_ZEROS_3[1:5, 3] = 1
correct_pattern_ZEROS_3[4, 1:3] = 1
correct_pattern_ZEROS_3.shape= (25,1)

#Training Pattern Zeros 4
correct_pattern_ZEROS_4= np.zeros((25,1))
correct_pattern_ZEROS_4.shape = (5,5)
correct_pattern_ZEROS_4[0, 2:4] = 1
correct_pattern_ZEROS_4[1:5, 1] = 1
correct_pattern_ZEROS_4[4, 1:3] = 1
correct_pattern_ZEROS_4[3, 3] = 1
correct_pattern_ZEROS_4[1:3, 4] = 1
correct_pattern_ZEROS_4.shape= (25,1)

#Training Pattern Zeros 5
correct_pattern_ZEROS_5= np.zeros((25,1))
correct_pattern_ZEROS_5.shape = (5,5)
correct_pattern_ZEROS_5[0, 1:4] = 1
correct_pattern_ZEROS_5[1:4, 0] = 1
correct_pattern_ZEROS_5[4, 1:4] = 1
correct_pattern_ZEROS_5[1:4, 4] = 1
correct_pattern_ZEROS_5.shape= (25,1)

#Training Pattern Zeros 6
correct_pattern_ZEROS_6= np.zeros((25,1))
correct_pattern_ZEROS_6.shape = (5,5)
correct_pattern_ZEROS_6[0, 0:4] = 1
correct_pattern_ZEROS_6[0:4, 0] = 1
correct_pattern_ZEROS_6[4, 0:4] = 1
correct_pattern_ZEROS_6[0:5, 4] = 1
correct_pattern_ZEROS_6[2,2] = 1
correct_pattern_ZEROS_6.shape= (25,1)


###END OF 0 PATTERNS###


###START OF MODEL TRAINING###



#Training Phase (1 to 5 (1 patterns)):

Desired_Output_Training_Set = 1 

#Training on Sample 1
Network_Output_1 = (np.dot(Weight_Vector.transpose(), Ones_Training_Pattern_1)) + Bias 
Weight_Vector, Bias = Weight_AND_Bias_Correction(Learning_Rate, Weight_Vector, Bias, Ones_Training_Pattern_1, Desired_Output_Training_Set, Network_Output_1)

#Training on Sample 2
Network_Output_2 = (np.dot(Weight_Vector.transpose(), Ones_Training_Pattern_2)) + Bias 
Weight_Vector, Bias = Weight_AND_Bias_Correction(Learning_Rate, Weight_Vector, Bias, Ones_Training_Pattern_2, Desired_Output_Training_Set, Network_Output_2)


#Training on Sample 3
Network_Output_3 = (np.dot(Weight_Vector.transpose(), Ones_Training_Pattern_3)) + Bias 
Weight_Vector, Bias = Weight_AND_Bias_Correction(Learning_Rate, Weight_Vector, Bias, Ones_Training_Pattern_3, Desired_Output_Training_Set, Network_Output_3)

#Training on Sample 4
Network_Output_4 = (np.dot(Weight_Vector.transpose(), Ones_Training_Pattern_4)) + Bias 
Weight_Vector, Bias = Weight_AND_Bias_Correction(Learning_Rate, Weight_Vector, Bias, Ones_Training_Pattern_4, Desired_Output_Training_Set, Network_Output_4)


#Training on Sample 5
# Network_Output_5 = (np.dot(Weight_Vector.transpose(), Ones_Training_Pattern_5)) + Bias
# Weight_Vector, Bias = Weight_AND_Bias_Correction(Learning_Rate, Weight_Vector, Bias, Ones_Training_Pattern_5, Desired_Output_Training_Set, Network_Output_5)


# #Training on Sample 6
# Network_Output_6 = (np.dot(Weight_Vector.transpose(), Ones_Training_Pattern_6)) + Bias
# Weight_Vector, Bias = Weight_AND_Bias_Correction(Learning_Rate, Weight_Vector, Bias, Ones_Training_Pattern_6, Desired_Output_Training_Set, Network_Output_6)


#END OF TRAINING PHASE ON ONES PATTERNS

#START OF TRAINING PHASE ON ZEROS PATTERNS


Desired_Output_Zeros = -1 

Network_Output_6 = (np.dot(Weight_Vector.transpose(), correct_pattern_ZEROS_1)) + Bias
Weight_Vector, Bias = Weight_AND_Bias_Correction(Learning_Rate, Weight_Vector, Bias, correct_pattern_ZEROS_1, Desired_Output_Zeros, Network_Output_6)

Network_Output_7 = (np.dot(Weight_Vector.transpose(), correct_pattern_ZEROS_2)) + Bias
Weight_Vector, Bias = Weight_AND_Bias_Correction(Learning_Rate, Weight_Vector, Bias, correct_pattern_ZEROS_2, Desired_Output_Zeros, Network_Output_7)

Network_Output_8 = (np.dot(Weight_Vector.transpose(), correct_pattern_ZEROS_3)) + Bias
Weight_Vector, Bias = Weight_AND_Bias_Correction(Learning_Rate, Weight_Vector, Bias, correct_pattern_ZEROS_3, Desired_Output_Zeros, Network_Output_8)

Network_Output_9 = (np.dot(Weight_Vector.transpose(), correct_pattern_ZEROS_4)) + Bias
Weight_Vector, Bias = Weight_AND_Bias_Correction(Learning_Rate, Weight_Vector, Bias, correct_pattern_ZEROS_4, Desired_Output_Zeros, Network_Output_9)

# Network_Output_10 = (np.dot(Weight_Vector.transpose(), correct_pattern_ZEROS_5)) + Bias
# Weight_Vector, Bias = Weight_AND_Bias_Correction(Learning_Rate, Weight_Vector, Bias, correct_pattern_ZEROS_5, Desired_Output_Zeros, Network_Output_10)



#END OF TRAINING PHASE ON ZEROS PATTERNS

print('Final Weight Vector Value:', Weight_Vector)
print('Final Bias Value:', Bias)


#Testing Phase
Testing_Output_1 = (np.dot(Weight_Vector.transpose(), correct_pattern_ZEROS_6)) + Bias

if (Testing_Output_1 > 0):
	print('This is a 1')
else:
	print('This is a 0')


Testing_Output_2 = (np.dot(Weight_Vector.transpose(), Ones_Training_Pattren_6)) + Bias

if (Testing_Output_2 > 0):
	print('This is a 1')
else:
	print('This is a 0')