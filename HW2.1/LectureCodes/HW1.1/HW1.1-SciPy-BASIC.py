import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from   scipy.optimize import minimize

#USER PARAMETERS
IPLOT=True
INPUT_FILE='weight.json'
FILE_TYPE="json"
DATA_KEYS=['x','is_adult','y']
OPT_ALGO='BFGS'

#UNCOMMENT FOR VARIOUS MODEL CHOICES (ONE AT A TIME)
model_type="logistic"; NFIT=4; xcol=1; ycol=2;
# model_type="linear";   NFIT=2; xcol=1; ycol=2; 
# model_type="logistic";   NFIT=4; xcol=2; ycol=0;

#READ FILE
with open(INPUT_FILE) as f:
	my_input = json.load(f)  #read into dictionary

#CONVERT INPUT INTO ONE LARGE MATRIX (SIMILAR TO PANDAS DF)
X=[];
for key in my_input.keys():
	if(key in DATA_KEYS): X.append(my_input[key])

#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
X=np.transpose(np.array(X))

#SELECT COLUMNS FOR TRAINING 
x=X[:,xcol];  y=X[:,ycol]

#EXTRACT AGE<18
if(model_type=="linear"):
	y=y[x[:]<18]; x=x[x[:]<18]; 

#COMPUTE BEFORE PARTITION AND SAVE FOR LATER
XMEAN=np.mean(x); XSTD=np.std(x)
YMEAN=np.mean(y); YSTD=np.std(y)

#NORMALIZE
x=(x-XMEAN)/XSTD;  y=(y-YMEAN)/YSTD; 

#PARTITION
f_train=0.8; f_val=0.2
rand_indices = np.random.permutation(x.shape[0])
CUT1=int(f_train*x.shape[0]); 
train_idx,  val_idx = rand_indices[:CUT1], rand_indices[CUT1:]
xt=x[train_idx]; yt=y[train_idx]; xv=x[val_idx];   yv=y[val_idx]

#MODEL
def model(x,p):
	if(model_type=="linear"):   return  p[0]*x+p[1]  
	if(model_type=="logistic"): return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.01))))

#SAVE HISTORY FOR PLOTTING AT THE END
iteration=0; iterations=[]; loss_train=[];  loss_val=[]

#LOSS FUNCTION
def loss(p,xt,yt):
	global iterations,loss_train,loss_val,iteration

	#TRAINING LOSS
	yp=model(xt,p) #model predictions for given parameterization p
	training_loss=(np.mean((yp-yt)**2.0))  #MSE

	#VALIDATION LOSS
	yp=model(xv,p) #model predictions for given parameterization p
	validation_loss=(np.mean((yp-yv)**2.0))  #MSE

	#WRITE TO SCREEN
	if(iteration==0):    print("iteration	training_loss	validation_loss") 
	if(iteration%25==0): print(iteration,"	",training_loss,"	",validation_loss) 
	
	#RECORD FOR PLOTING
	loss_train.append(training_loss); loss_val.append(validation_loss)
	iterations.append(iteration); iteration+=1

	return training_loss


#INITIAL GUESS
po=np.random.uniform(0.1,1.,size=NFIT)
NDIM = 4

def pass_para(xt,yt,po,loss, tol = 1e-10, LR = 0.001, method = 'batch', algo = 'GD'):
    dx = 0.001
    t = 0
    tmax = 50000
    while t<=tmax:
        t +=1
        df_dx = np.zeros(NDIM)
        for i in range(0,NDIM):
            dX = np.zeros(NDIM)
            dX[i] = dx
            xm1 = po-dX
            df_dx[i] = (loss(po,xt,yt)-loss(xm1,xt,yt))/dx
            xip1 = po-LR*df_dx
        if(t%100==0):
            df = np.mean(np.absolute(loss(xip1,xt,yt)-loss(po,xt,yt)))
            print(t,"	",po,"	","	",loss(po,xt,yt))
            if(df<tol):
                print("STOPPING CRITERION MET (STOPPING TRAINING)")
                break
        po=xip1
    return po

def momentum(xt,yt,po,loss, tol = 1e-10, LR = 0.001, method = 'batch', algo = 'GD'):
    dx = 0.001
    t = 0
    tmax = 50000
    mum = 0.001
    while t<=tmax:
        t +=1
        df_dx = np.zeros(NDIM)
        for i in range(0,NDIM):
            dX = np.zeros(NDIM)
            dX[i] = dx
            xm1 = po-dX
            df_dx[i] = (loss(po,xt,yt)-loss(xm1,xt,yt))/dx
            xip1 = po-LR*df_dx+dx*mum
        if(t%100==0):
            df = np.mean(np.absolute(loss(xip1,xt,yt)-loss(po,xt,yt)))
            print(t,"	",po,"	","	",loss(po,xt,yt))
            if(df<tol):
                print("STOPPING CRITERION MET (STOPPING TRAINING)")
                break
        po=xip1
    return po

def optimizer(loss, tol = 1e-10, LR=0.001, method = 'batch', algo = 'GD'):
    global po, NDIM
    if method == 'batch' and algo == 'GD':
        dx=0.001
        t=0
        tmax=100000
        while t<=tmax:
            t +=1
            df_dx=np.zeros(NDIM)
            for i in range(0,NDIM):
                dX=np.zeros(NDIM)
                dX[i]=dx
                xm1=po-dX
                df_dx[i]=(loss(po,xt,yt)-loss(xm1,xt,yt))/dx
                xip1=po-LR*df_dx
            if(t%100==0):
                df=np.mean(np.absolute(loss(xip1,xt,yt)-loss(po,xt,yt)))
                print(t,"	",po,"	","	",loss(po,xt,yt))
                if(df<tol):
                    print("STOPPING CRITERION MET (STOPPING TRAINING)")
                    break
            po=xip1
        return po
    
    if method == 'mini-batch' and algo == 'GD':
        n = len(xt)
        xt1,xt2 = xt[:n//2], xt[n//2:]
        yt1,yt2 = yt[:n//2], yt[n//2:]
        p1 = pass_para(xt1,yt1,po,loss, tol = 1e-10, LR = 0.001, method = 'batch', algo = 'GD')
        p2 = pass_para(xt2,yt2,p1,loss, tol = 1e-10, LR = 0.001, method = 'batch', algo = 'GD')
        return p2
        
    if method == 'SGD' and algo == 'GD':
        for i in range(len(xt)):
            po = pass_para(xt[i],yt[i],po,loss, tol = 1e-10, LR = 0.001, method = 'batch', algo = 'GD')
        return po
            
        
            
        


#TRAIN MODEL USING SCIPY MINIMIZ 
#res = optimizer(loss, tol = 1e-10, LR = 0.001, method = 'batch', algo = 'GD');
#res  = optimizer(loss, tol = 1e-10, LR=0.001, method = 'mini-batch', algo = 'GD')
#res = optimizer(loss, tol = 1e-10, LR=0.001, method = 'SGD', algo = 'GD')
#res = optimizer(loss, tol = 1e-10, LR=0.005, method = 'SGD', algo = 'GD')
#res = optimizer(loss, tol = 1e-10, LR=0.003, method = 'SGD', algo = 'GD')
#res = optimizer(loss, tol = 1e-10, LR=0.003, method = 'batch', algo = 'GD')
#res = optimizer(loss, tol = 1e-10, LR=0.005, method = 'batch', algo = 'GD')
#res = optimizer(loss, tol = 1e-10, LR=0.003, method = 'mini-batch', algo = 'GD')
res = momentum(xt,yt,po,loss, tol = 1e-10, LR = 0.001, method = 'batch', algo = 'GD')
print("OPTIMAL PARAM:",res)

#PREDICTIONS
xm=np.array(sorted(xt))
yp=np.array(model(xm,res))

#UN-NORMALIZE
def unnorm_x(x): 
	return XSTD*x+XMEAN  
def unnorm_y(y): 
	return YSTD*y+YMEAN 

#FUNCTION PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(unnorm_x(xt), unnorm_y(yt), 'o', label='Training set')
	ax.plot(unnorm_x(xv), unnorm_y(yv), 'x', label='Validation set')
	ax.plot(unnorm_x(xm),unnorm_y(yp), '-', label='Model')
	plt.xlabel('x', fontsize=18)
	plt.ylabel('y', fontsize=18)
	plt.legend()
	plt.show()

#PARITY PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(model(xt,res), yt, 'o', label='Training set')
	ax.plot(model(xv,res), yv, 'o', label='Validation set')
	plt.xlabel('y predicted', fontsize=18)
	plt.ylabel('y data', fontsize=18)
	plt.legend()
	plt.show()

#MONITOR TRAINING AND VALIDATION LOSS  
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(iterations, loss_train, 'o', label='Training loss')
	ax.plot(iterations, loss_val, 'o', label='Validation loss')
	plt.xlabel('optimizer iterations', fontsize=18)
	plt.ylabel('loss', fontsize=18)
	plt.legend()
	plt.show()

# I HAVE WORKED THROUGH THIS EXAMPLE AND UNDERSTAND EVERYTHING THAT IT IS DOING