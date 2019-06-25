from utilis1 import json_read
import matplotlib.pyplot as plt
import argparse
 
def hyper_params(args):
    obj=json_read(args.result)
    name=obj[0]['Graph']
    hidden=obj[0]['Hidden layer']
    batch=obj[0]['Batch size']
    rate=obj[0]['Learning rate']
    return name,hidden,batch,rate

def plot_graph(args,x,y):
    fig=plt.figure(figsize=(10,5))
    names=range(len(x))
    ax = fig.add_subplot(111)
    ax.scatter(x,y,label='Data')
    plt.plot(names, y,label='line',color='red')
    plt.xticks(names, x)
    plt.xlabel('number of epochs')
    plt.ylim(0.7,1)
    plt.ylabel('ROC AUC score')
    name,hid,batch,rate=hyper_params(args)
    plt.title('Name: {}. Hidden: {}. Batch: {}. LR: {}.'.format(name,hid,batch,rate))
    ax.legend()
    plt.show()

def json2list(args):
    
    obj=json_read(args.result)
    roc=[]
    epoch=[]
    for i in range(1,len(obj)):
        val=obj[i]['epoch']
        epoch.append(val)
        val= obj[i]['ROC AUC score']
        roc.append(val)
    return epoch,roc
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result",
                        nargs = "?",
                        default = "./output/result.txt",
                        help = "neural network result file path.")
    args=parser.parse_args()
    x,y=json2list(args)
    plot_graph(args,x,y)
