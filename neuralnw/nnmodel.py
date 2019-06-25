from utilis1 import json_dumper
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


def nn_result(epoch,loss,f1,roc,acc,args):

	data = dict()
	data["epoch"] = epoch
	data["loss"] = loss
	data["accuracy"] = acc
	data["F1-score"] = f1
	data["ROC AUC score"] = roc
	json_dumper(data,args.result)

def standardize(predictions):
	max_v = predictions.max()
	std_pred = predictions/max_v
	return std_pred

def f1_(y_preds, y_targets):
	y_true = y_targets.numpy()
	y_pred = y_preds.numpy() 
	return(f1_score(y_true, y_pred, average='micro'))

def roc_auc(y_preds, y_targets):
	y_true = y_targets.numpy()
	y_pred = y_preds.detach().numpy()
	return roc_auc_score(y_true, y_pred,average='micro')

def neuralnw(G,train_loader,val_loader,args):
	n_in =len(G.nodes())
	n_h = args.hidden
	n_out = len(G.nodes())
	model = nn.Sequential(nn.Linear(n_in, n_h),
    nn.ReLU(),
    nn.Linear(n_h, n_out),
    nn.Sigmoid())
	#Construct the loss function
	criterion = torch.nn.MSELoss()
	# Construct the optimizer 
	optimizer = torch.optim.Adam(model.parameters(), args.lr)

	
	batch_size = args.batch
	num_epochs=args.epoch
	# Gradient Descent
	for epoch in range(num_epochs):
	    model.train()
	    for i, (in_,fin_) in enumerate(train_loader):
	    	in_ = in_.view(-1,len(G.nodes())).requires_grad_(True)
	    	# Clear gradients w.r.t. parameters

	    	optimizer.zero_grad()
	    	# Forward pass: Compute predicted y by passing x to the model
	    	y_pred = model(in_)
	    	# Compute and print loss
	    	loss = criterion(y_pred, fin_)

	    	# perform a backward pass (backpropagation)
	    	loss.backward()
	    	# Update the parameter
	    	optimizer.step()
	    model.eval()
	    acc=0
	    total=0
	    f1_score=0
	    roc=0
	    for i, (in_, fin_) in enumerate(val_loader):
	    	# Forward pass only to get logits/output
	    	predictions = model(in_)
	    	predictions=standardize(predictions)
	    	#use probalistic predictions for roc score
	    	roc += roc_auc(predictions,fin_)
	    	threshold = Variable(torch.Tensor([args.threshold]))
	    	predictions = (predictions > threshold).float() * 1
	    	#use binary values for f1 score
	    	f1_score += f1_(predictions,fin_)
	    	accuracy_scores = (predictions==fin_).type(torch.FloatTensor)
	    	acc += torch.sum(accuracy_scores)
	    	total += fin_.size(1)*batch_size

	    nn_result(epoch,loss.item(),f1_score/(i+1),roc/(i+1),100*acc.item()/total,args)
	    print('Epoch: {}. Loss: {}. Accuracy: {}. F1 Score: {}.'.format(epoch, loss.item(),100*acc/total,f1_score/(i+1)))
	    print('Roc score:',roc/(i+1))
