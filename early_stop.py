import os
import string
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pickle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch
import torch.optim as optim
from numpy.random import permutation
from datetime import datetime
from sklearn.model_selection import KFold

def timediff(t1,t2):
	t1=(t1 + '000')
	t2=(t2 + '000')
	day1=datetime.strptime(t1, "%d:%m:%y:%H:%M:%S:%f")
	day2=datetime.strptime(t2, "%d:%m:%y:%H:%M:%S:%f")
	sec = (day2-day1).total_seconds()
	return(sec)


def extract_features(s,path):
	save_loc = os.path.join('./Features/',s+'/')
	if not os.path.isdir(save_loc):
		os.makedirs(save_loc)
	fpaths = []
	files = sorted(os.listdir(path+s))
	for f in files:
		fpaths.append(os.path.join(path+s,f))
	for index,f in enumerate(sorted(fpaths)):
		lines = [line.rstrip('\t\n') for line in open(f)]
		f_list = [event.split('\t') for event in lines]
		##########################################################
		alp = list(string.ascii_lowercase)
		alp2 = list(map(''.join,itertools.combinations(string.ascii_lowercase,2)))
		##########################################################
	
		KeyUps = [x for x in f_list if 'KeyUp' in x and len(x)==3]
		KeyDowns = [x for x in f_list if 'KeyDown' in x and len(x)==3]

		###########################################################

		tups =  [item[2] for item in KeyUps]# if item[1] in alp]
		tdowns =  [item1[2] for item1 in KeyDowns]# if item1[1] in alp]
		letterup =  [item[1].upper() for item in KeyUps]# if item[1] in alp]
		letterdown = [item1[1].upper() for item1 in KeyDowns]# if item1[1] in alp]
			
		#############################################################
		features = []
		for i in range(0,len(tups)-1):
			t = i
			t1 = tdowns[i]
			if letterup[t] != letterdown[i]:
				j = i
				if i == len(tups)-1:
					j = 0
				while j<len(tups)-1 and letterdown[i]!= letterup[j] and i!=len(tups)-1:
					j = j+1
				tj = tups[j]
				k = i
				if i == 0:
					k = len(tups)-1
				while k>=1 and letterdown[i]!= letterup[k] and i!=0:
					k = k-1
				tk = tups[k]
				if timediff(t1,tk)>0 and timediff(t1,tj)>0 :
					if abs(j-i)<abs(i-k):
						t = j
					else:
						t = k
				elif timediff(t1,tk)<0 :
					t = j
				else:
					t = k
			t2 = tups[t]
			if i!=len(tups)-1:
				t3 = tdowns[i+1]
				latency = timediff(t1,t3)
				lat = letterdown[i]+letterdown[i+1],latency
				features.append(lat)
			hold_time = timediff(t1,t2)
			hold = letterdown[i],hold_time
			features.append(hold)
		with open(save_loc+files[index], "wb") as fp:   #Pickling
			pickle.dump(features, fp)

def features2array(K):
	S = np.zeros((26,1))
	T = np.zeros((26*26,1))
	freq1 = np.zeros((26,1))
	freq2 = np.zeros((26*26,1))
	for i in range(len(K)):
		if K[i][0] in alp:
			pos = alp.index(K[i][0])
			freq1[pos,0] += 1
			S[pos,0] += K[i][1]
		if K[i][0] in alp2:
			pos2 = alp2.index(K[i][0])
			freq2[pos2,0] += 1
			T[pos2,0] += K[i][1]
	for i in range(26):
		if freq1[i,0] != 0:
			S[i,0] /= freq1[i,0]
	for i in range(26*26):
		if freq2[i,0] != 0:
			T[i,0] /= freq2[i,0]
	F = np.concatenate((S,T),axis = 0)
	return F

def encode(s,fpath,ipath):
	filepath = os.path.join(fpath,s + '/')
	count = 0
	for x in sorted(os.listdir(filepath)):
		with open(filepath + x,'rb') as f:
			K = pickle.load(f)
		F = features2array(K).flatten()
		save_loc = os.path.join(ipath,s+'/')
		if not os.path.isdir(save_loc):
			os.makedirs(save_loc)
		with open(save_loc + x,'wb') as f:
			pickle.dump(F,f)

def newfeatures(s,path):
	if len(os.listdir(path+s))>=25:
		return
	else:
		print(s)
		newfile = len(os.listdir(path+ s))+1
		with open(path + s + '/1.txt','rb') as f:
			K = pickle.load(f)
		sum = K
		with open(path + s + '/2.txt','rb') as f:
			K = pickle.load(f)
		sum += K
		with open(path + s + '/3.txt','rb') as f:
			K = pickle.load(f)
		sum += K
		count = 3
		while len(os.listdir(path+s))<25:
			count += 1
			with open(path + s + '/' + str(newfile) +'.txt','wb') as f:
				avg = sum/3
				newfile += 1
				pickle.dump(avg,f)
			with open(path + s + '/' + str(count) + '.txt','rb') as f:
				K = pickle.load(f)
			sum += K
			with open(path + s + '/' + str(count-3) + '.txt','rb') as f:
				K = pickle.load(f)
			sum -= K

def get_legal_data(indices,path):
	legal_data = np.zeros((len(indices),25,702))
	for index in range(len(indices)):
		u = Users[indices[index]]
		filepath = os.path.join(path,u+'/')
		for id,f in enumerate(sorted(os.listdir(filepath))):
			if id>24:
				break
			with open(filepath + f,'rb') as file:
				K = pickle.load(file)
			legal_data[index,id,:] = K
	return legal_data   

def get_imposters(indices,path):
	imposters = np.zeros((len(indices),25,702)) 
	for index in range(len(indices)):
		u = Users[indices[index]]
		filepath = os.path.join(path,u+'/')
		for id,f in enumerate(sorted(os.listdir(filepath))):
			if id>24:
				break
			with open(filepath + f,'rb') as file:
				K = pickle.load(file)
			imposters[index,id,:] = K
	return imposters

def train_test_split(legal_data,imposters,N,M):
	X_train = np.zeros((200,702))
	y_train = np.zeros((200,1))
	X_test = np.zeros((50,702))
	y_test = np.zeros((50,1))
	count = 0
	for i in range(N):
		X_train[count*20:20*(count+1),:] = legal_data[i,0:20,:]
		X_test[count*5:5*(count+1),:] = legal_data[i,20:25,:]
		y_test[count*5:5*(count+1),:] = 1
		y_train[count*20:20*(count+1),:] = 1
		count += 1
	for i in range(M):
		X_train[count*20:20*(count+1),:] = imposters[i,0:20,:]
		X_test[count*5:5*(count+1),:] = imposters[i,20:25,:]
		count += 1
	return X_train, y_train, X_test, y_test

class ANN(nn.Module):

	def __init__(self):
		super(ANN, self).__init__()
		# an affine operation: y = Wx + b
		self.fc1 = nn.Linear(702, 100)  # 6*6 from image dimension
		self.fc2 = nn.Linear(100,40)
		self.fc3 = nn.Linear(40, 2)

	def forward(self, x):
		# Max pooling over a (2, 2) window
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = torch.sigmoid(self.fc3(x))
		return x

def init_weights(m):
	if type(m) == nn.Linear:
		nn.init.xavier_uniform_(m.weight)
		m.bias.data.fill_(0.01)

def train_model(X_train, y_train, X_test, y_test):

	TrainDataset = torch.utils.data.TensorDataset(torch.tensor(X_train).type('torch.FloatTensor'),torch.tensor(y_train).type(torch.LongTensor))
	TestDataset = torch.utils.data.TensorDataset(torch.tensor(X_test).type('torch.FloatTensor'),torch.tensor(y_test).type(torch.LongTensor))
	batch_size = 4
	train_loader = torch.utils.data.DataLoader(dataset=TrainDataset,batch_size=batch_size,shuffle=True,drop_last = True)
	test_loader = torch.utils.data.DataLoader(dataset=TestDataset,batch_size=batch_size,shuffle=True,drop_last = True)

	model = ANN()
	model.apply(init_weights)
	criterion = nn.CrossEntropyLoss()
	learning_rate = 0.01
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


	TrainAcc = []
	ValAcc = []
	n_epochs_stop = 20
	epochs_no_improve = 0
	early_stop = False

	val_loss = 0
	min_val_loss = np.Inf

	iter = 0
	for epoch in range(120):
		train_loss = 0
		train_correct = 0
		train_total = 0
		model.train()
		for i, (images, labels) in enumerate(train_loader):
			images = images.view(-1, 702).requires_grad_()
			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels.flatten())
			_, predicted = torch.max(outputs,1)
			train_total += labels.size(0)
			train_correct += (predicted == labels.flatten()).sum()
			train_loss += loss.item()
			loss.backward()
			optimizer.step()
			iter += 1
		train_accuracy = 100 * train_correct.item()/train_total
		model.eval()
		correct = 0
		total = 0
		# Iterate through test dataset
		with torch.no_grad():
			for xtest,ytest in test_loader:
				# Load images with gradient accumulation capabilities
					xtest = xtest.view(-1,702)
					# Forward pass only to get logits/output
					output = model(xtest)
					loss = criterion(output,ytest.flatten())
					val_loss += loss.item()
					# Get predictions from the maximum value
					_, predicted = torch.max(output,1)
					# Total number of labels
					total += ytest.size(0)
					# Total correct predictions
					correct += (predicted == ytest.flatten()).sum()
			val_loss = val_loss / len(test_loader)
			if val_loss < min_val_loss:
				 epochs_no_improve = 0
				 min_val_loss = val_loss
			else:
				epochs_no_improve += 1
				if epoch > 19 and epochs_no_improve == n_epochs_stop:
					print('Early stopping!' )
					print("Stopped")
					early_stop = True
					break
		val_accuracy = 100 * correct.item()/ total
		# Print Loss
		if epoch%20==0:
			print('Iteration: %s. Train Accuracy: %.4f. Validation Accuracy: %.4f'%(epoch, train_accuracy, val_accuracy))
		TrainAcc.append(train_accuracy)
		ValAcc.append(val_accuracy)
	return TrainAcc, ValAcc

def plotTrainingCurves(TrainAcc,ValAcc):
	
	T = np.zeros((5,120))
	V = np.zeros((5,120))
	e = np.arange(120)

	for i in range(5):
		T[i] = TrainAcc[i]
		V[i] = ValAcc[i]

	m0 = T.mean(axis = 1)
	s0 = T.std(axis = 1)
	m1 = V.mean(axis = 1)
	s1 = V.std(axis = 1)
	posV = np.argmax(m1)

	fig, ax1 = plt.subplots()
	color = 'tab:red'
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Training_Accuracy', color=color)
	ax1.plot(e,T[posV], color=color)
	ax1.tick_params(axis='y', labelcolor=color)
	fig.tight_layout()

	fig1, ax1 = plt.subplots()
	color = 'tab:red'
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Validation_Accuracy', color=color)
	ax1.plot(e,V[posV], color=color)
	ax1.tick_params(axis='y', labelcolor=color)
	fig.tight_layout()
	plt.show()

def plotCrossValCurves(TrainAcc,ValAcc):

	T = np.zeros((5,120))
	V = np.zeros((5,120))
	e = np.arange(120)

	for i in range(5):
		T[i] = TrainAcc[i]
		V[i] = ValAcc[i]

	m0 = T.mean(axis = 0)
	s0 = T.std(axis = 0)
	m1 = V.mean(axis = 0)
	s1 = V.std(axis = 0)

	fig, ax1 = plt.subplots()
	color = 'tab:red'
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Mean Train Accuracy', color=color)
	ax1.plot(e,m0,label='mean_train', color=color)
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()
	color = 'tab:blue'
	ax2.set_ylabel('Mean Valid Accuracy', color=color)  # we already handled the x-label with ax1
	ax2.plot(e,m1,label='mean_valid', color=color)
	ax2.tick_params(axis='y', labelcolor=color)

	ax1.fill_between(e, m0+s0, m0-s0, facecolor='tab:red', alpha=0.5)
	ax2.fill_between(e, m1+s1, m1-s1, facecolor='tab:blue', alpha=0.5)

	ax1.set_title(r'Mean and Standard Deviation in Accuracies')

	ax1.grid()
	ax2.grid()
	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.show()

root = os.getcwd()
path = os.path.join(root,"Data/")
fpath = os.path.join(root,"Features/")
ipath = os.path.join(root,"TrainData/")
subjects = sorted(os.listdir(path))
Users = subjects

for s in subjects:
	extract_features(s,path)

alp = list(string.ascii_uppercase)
alp2 = list(map(''.join,itertools.combinations(string.ascii_uppercase,2)))

for s in subjects:
	encode(s,fpath,ipath)

for s in subjects:
	newfeatures(s,ipath)

indices = permutation(10)
kf = KFold(n_splits=5,shuffle = True)
kf.get_n_splits(indices)
TrainAcc = []
ValAcc = []
for i,(train_index, test_index) in enumerate(kf.split(indices)):
	print("Validation Number: %s"%(i))
	print("Authentic Users : ",test_index)
	print("imposters : ",train_index)
	legal_data = get_legal_data(test_index,ipath)
	imposters = get_imposters(train_index,ipath)
	N = len(train_index)
	M = len(test_index)
	print(legal_data.shape,imposters.shape)
	X_train, y_train, X_test, y_test = train_test_split(legal_data,imposters,M,N)
	T,V = train_model(X_train,y_train,X_test,y_test)
	TrainAcc.append(T)
	ValAcc.append(V)

# plotTrainingCurves(TrainAcc,ValAcc)
# plotCrossValCurves(TrainAcc,ValAcc)