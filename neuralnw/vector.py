import pandas as pd
from utilis1 import json_read
global seed
global active

def func1(f):
	global seed
	if f in seed:
		return 1
	else:
		return 0
def func2(f):
	global active
	if f in active:
		return 1
	else:
		return 0

def json2vec(args,G):
	"""
	Method to convert seed and activations to input and output binary vector.
	:param path: path of json file.
	:return param csv_vec: csv file containing vectors
	"""
	global seed
	global active
	obj=json_read(args.json)
	l=[]
	for i in range(1,len(obj)):
	    in_vec=list(G.nodes())
	    fin_vec=list(G.nodes())
	    seed=obj[i]['seed set'] 
	    active=obj[i]['sequence']
	    inp=list(map(func1,in_vec))
	    fin=list(map(func2,fin_vec))
	    inp.extend(fin)
	    l.append(inp)
	df = pd.DataFrame(l,columns = [str(x) for x in range(0,2*len(G.nodes()))]) 
	csv_vec = df.to_csv (args.vector, index = None, header=True) 
	return csv_vec
	    


