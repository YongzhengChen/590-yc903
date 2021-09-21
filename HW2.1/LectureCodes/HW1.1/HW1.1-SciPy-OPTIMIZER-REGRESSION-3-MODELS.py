#--------------------------------
# UNIVARIABLE REGRESSION EXAMPLE
#    -USING SciPy FOR OPTIMIZATION
#--------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from   scipy.optimize import minimize


#------------------------
#CODE PARAMETERS
#------------------------

#USER PARAMETERS
IPLOT=True
INPUT_FILE='weight.json'
FILE_TYPE="json"
DATA_KEYS=['x','is_adult','y']

#HYPER-PARAM
OPT_ALGO='CG'

#------------------------
#DATA CLASS
#------------------------

class DataClass:

    #INITIALIZE
	def __init__(self,FILE_NAME):

		if(FILE_TYPE=="json"):

			#READ FILE
			with open(FILE_NAME) as f:
				self.input = json.load(f)  #read into dictionary

			#CONVERT INPUT INTO ONE LARGE MATRIX 
				#SIMILAR TO PANDAS DF   
			X=[];
			for key in self.input.keys():
				if(key in DATA_KEYS):
					X.append(self.input[key])

			#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
			self.X=np.transpose(np.array(X))

			#TAKE MEAN AND STD DOWN COLUMNS (I.E DOWN SAMPLE DIMENSION)
			self.XMEAN=np.mean(self.X,axis=0)
			self.XSTD=np.std(self.X,axis=0)

		else:
			raise ValueError("REQUESTED FILE-FORMAT NOT CODED");

	def report(self):
		print("--------DATA REPORT--------")
		print("X shape:", self.X.shape)
		print("X means:",np.mean(self.X,axis=0))
		print("X stds:",np.std(self.X,axis=0))
		print("X examples")
		#PRINT FIRST 5 SAMPLES
		for i in range(0,self.X.shape[1]):
			print("X column ",i,": ",self.X[0:5,i])

	def partition(self,f_train=0.825, f_val=0.15, f_test=0.025):
		#f_train=fraction of data to use for training

		#TRAINING: 	 DATA THE OPTIMIZER "SEES"
		#VALIDATION: NOT TRAINED ON BUT MONITORED DURING TRAINING
		#TEST:		 NOT MONITORED DURING TRAINING (ONLY USED AT VERY END)
		if(f_train+f_val+f_test != 1.0):
			raise ValueError("f_train+f_val+f_test MUST EQUAL 1");

		#PARTITION DATA
		rand_indices = np.random.permutation(self.X.shape[0]) #randomize indices
		CUT1=int(f_train*self.X.shape[0]); 
		CUT2=int((f_train+f_val)*self.X.shape[0]); 
		self.train_idx, self.val_idx, self.test_idx = rand_indices[:CUT1], rand_indices[CUT1:CUT2], rand_indices[CUT2:]

	def plot_xy(self,col1=1,col2=2,xla='x',yla='y'):
		if(IPLOT):
			fig, ax = plt.subplots()
			FS=18   #FONT SIZE
			ax.plot(self.X[:,col1], self.X[:,col2],'o') #,c=data['y'], cmap='gray')
			plt.xlabel(xla, fontsize=FS)
			plt.ylabel(yla, fontsize=FS)
			plt.show()

	def normalize(self):
		self.X=(self.X-self.XMEAN)/self.XSTD #/3.


#------------------------
#MAIN 
#------------------------

#INITIALIZE DATA OBJECT 
Data=DataClass(INPUT_FILE)

#BASIC DATA PRESCREENING
Data.report()
Data.partition()
Data.normalize()
Data.report()

Data.plot_xy(1,2,'age (years)','weight (lb)')
Data.plot_xy(2,0,'weight (lb)','is_adult')


#------------------------
#DEFINE MODEL
#------------------------

def model(x,p):
	global model_type
	if(model_type=="linear"): return p[0]*x+p[1]  
	if(model_type=="logistic"): return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))

#UN-NORMALIZE
def unnorm(x,col): 
	# print(Data.XSTD[col])
	return Data.XSTD[col]*x+Data.XMEAN[col] 


#------------------------
#DEFINE LOSS FUNCTION
#------------------------
iteration=0;

#SAVE HISTORY FOR PLOTTING AT THE END
iterations=[]; loss_train=[];  loss_val=[]

def loss(p):
	global iteration,iterations,loss_train,loss_val
	global xt,yt,xv,yv

	#TRAINING LOSS
	yp=model(xt,p) #model predictions for given parameterization p
	training_loss=(np.mean((yp-yt)**2.0))  #MSE

	#VALIDATION LOSS
	yp=model(xv,p) #model predictions for given parameterization p
	validation_loss=(np.mean((yp-yv)**2.0))  #MSE

	#WRITE TO SCREEN
	if(iteration%25==0):
		print(iteration,training_loss,validation_loss) #,p)

	#SAVE
	loss_train.append(training_loss)
	loss_val.append(validation_loss)
	iterations.append(iteration)

	iteration+=1

	return training_loss


#------------------------
#FIT 3 MODELS
#------------------------

for my_model in [1 ,2, 3]:

	# 1=linear regression (age vs weight)(age<18)
	# 2=logistic reg (age vs weight)
	# 3=logistic reg (weight vs is_adule)

	print('--------------------')
	print(my_model)
	print('--------------------')

	if(my_model==1):
		model_type="linear";   xcol=1; ycol=2; NFIT=2
	if(my_model==2):
		model_type="logistic"; xcol=1; ycol=2; NFIT=4
	if(my_model==3):
		model_type="logistic"; xcol=2; ycol=0; NFIT=4 
		#NFIT --> (NUMBER OF FITTING PARAMETERS)

	#RANDOM INITIAL GUESS FOR FITTING PARAMETERS
	po=np.random.uniform(0.5,1.,size=NFIT)

	#SELECT DATA FOR GIVEN MODEL
	#TRAINING
	xt=Data.X[:,xcol][Data.train_idx]	
	yt=Data.X[:,ycol][Data.train_idx] 
	#VALIDATION
	xv=Data.X[:,xcol][Data.val_idx]	
	yv=Data.X[:,ycol][Data.val_idx] 
	#TEST
	xtest=Data.X[:,xcol][Data.test_idx]	
	ytest=Data.X[:,ycol][Data.test_idx] 

	#EXTRACT AGE<18
	if(my_model==1):
		#NEED TO CONVERT 18 INTO NORMALIZED SPACE
		max_age=(18.-Data.XMEAN[xcol])/Data.XSTD[xcol]
		yt=yt[xt[:]<max_age]; xt=xt[xt[:]<max_age]; 
		yv=yv[xv[:]<max_age]; xv=xv[xv[:]<max_age]; 
		ytest=ytest[xtest[:]<max_age]
		xtest=xtest[xtest[:]<max_age]; 

	#TRAIN MODEL USING SCIPY MINIMIZ 
	res = minimize(loss, po, method=OPT_ALGO, tol=1e-15)
	popt=res.x
	print("OPTIMAL PARAM:",popt)

	#PREDICTIONS 
	xm=np.array(sorted(xt))
	yp=np.array(model(xm,popt))

	#FUNCTION PLOTS
	if(IPLOT):
		fig, ax = plt.subplots()
		ax.plot(unnorm(xt,xcol), unnorm(yt,ycol), 'o', label='Training set')
		ax.plot(unnorm(xv,xcol), unnorm(yv,ycol), 'x', label='Validation set')
		ax.plot(unnorm(xtest,xcol), unnorm(ytest,ycol), '*', label='Test set')
		ax.plot(unnorm(xm,xcol),unnorm(yp,ycol), '-', label='Model')
		plt.xlabel('x', fontsize=18)
		plt.ylabel('y', fontsize=18)
		plt.legend()
		plt.show()

	#PARITY PLOTS
	if(IPLOT):
		fig, ax = plt.subplots()
		ax.plot(model(xt,popt), yt, 'o', label='Training set')
		ax.plot(model(xv,popt), yv, 'o', label='Validation set')
		# ax.plot(yt, yt, '-', label='y_pred=y_data')

		plt.xlabel('y predicted', fontsize=18)
		plt.ylabel('y data', fontsize=18)
		plt.legend()
		plt.show()

	#MONITOR TRAINING AND VALIDATION LOSS  
	if(IPLOT):
		fig, ax = plt.subplots()
		#iterations,loss_train,loss_val
		ax.plot(iterations, loss_train, 'o', label='Training loss')
		ax.plot(iterations, loss_val, 'o', label='Validation loss')
		plt.xlabel('optimizer iterations', fontsize=18)
		plt.ylabel('loss', fontsize=18)
		plt.legend()
		plt.show()

# I HAVE WORKED THROUGH THIS EXAMPLE AND UNDERSTAND
EVERYTHING THAT IT IS DOING
