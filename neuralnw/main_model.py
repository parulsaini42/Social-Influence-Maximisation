from parse1 import parameter_parser
from utilis1 import read_graph
from vector import json2vec
from feednn import CustomDataset
from nnmodel import neuralnw
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import pandas as pd


def Dataloader(dataset,G):
    print('Data set loaded')
    lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]
    trainset, valset = random_split(dataset,lengths)

    train_loader = DataLoader(trainset,batch_size=100,shuffle=True)
    val_loader = DataLoader(valset,batch_size=100,shuffle=True)
    print('calling nn')
    neuralnw(G,train_loader,val_loader)

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
    vec=pd.read_csv(csv_path, engine='python')
    dataset = CustomDataset(args,G,vec)
    Dataloader(dataset,G)
    

if __name__ == "__main__":
    args = parameter_parser()
    initialize(args)





