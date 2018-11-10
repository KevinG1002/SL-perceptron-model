
import numpy as np

def correct_pattern_generator(): 
	correct_pattern = np.zeros((25,1))
	for i in range(len(correct_pattern)):
 		if (i  == 2 or  i == 7 or i==12 or i==17 or i==22):
 			correct_pattern[i] = 1
	return correct_pattern
def Desired_Output_Conversion(Training_Model):
	for i in range (len(Training_Model)):
		if Training_Model[i] == 0:
			Training_Model[i] = -1
	return Training_Model

def Weight_AND_Bias_Correction(Learning_Rate, Weight_Vector, Bias, Input_Vector, Desired_Output, Network_Output):
	Delta_X= np.multiply(Learning_Rate*(Desired_Output - Network_Output), Input_Vector)
	Weight_Vector = np.add(Weight_Vector, Delta_X)
	Bias = Bias + Learning_Rate*(Desired_Output - Network_Output) 
	return Weight_Vector, Bias 

