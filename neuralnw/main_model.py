from parse1 import parameter_parser
from utilis1 import read_graph,json_dumper
from vector import json2vec
from feednn import CustomDataset
from nnmodel import neuralnw
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import pandas as pd

def info(vector,args):
    data = dict()
    data["Graph"] = args.input
    data["Sample size"] = len(vector)
    data["Epochs"] = args.epoch
    data["Batch size"] = args.batch
    data["Hidden layer"] = args.hidden
    data["Threshold"] = args.threshold
    data["Learning rate"] = args.lr
    json_dumper(data,args.result)

def Dataloader(dataset,G,args):
    print('Data set loaded')
    lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]
    trainset, valset = random_split(dataset,lengths)

    train_loader = DataLoader(trainset,batch_size=args.batch,shuffle=True)
    val_loader = DataLoader(valset,batch_size=args.batch,shuffle=True)
    print('calling nn')
    neuralnw(G,train_loader,val_loader,args)

def initialize(args):
    """
    Method to run the model.
    :param args: Arguments object.
    """
    print('graph')

    G = read_graph(args.input)
    file_path=Path(args.vector)
    if file_path.is_file():
        csv_path=args.vector
        print('vector file already exists')
    else:
        #problem
        csv_path= json2vec(args,G)
    vec=pd.read_csv(csv_path)
    info(vec,args)
    dataset = CustomDataset(args,G,vec)
    Dataloader(dataset,G,args)
    

if __name__ == "__main__":
    args = parameter_parser()
    initialize(args)



