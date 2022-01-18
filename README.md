# Social-Influence-Maximisation

The project aims to predict and maximize social media influence. The model based approach involves training the neural network using simulations to predict active node set from a given seed set. The simulations are produced using propagation models as discussed in further sections. Further LSTM based sequence to sequence models that are based on the encoder decoder architecture can be used to predict the sequence depicting the spread and extracting useful information about the network structure which is unknown. The aim of the project is to analyse social networks whose structure is not explicitly known. In case of most of the real life networks like twitter network we don't know the structure of the graph. For example we know that a tweet has been re-tweeted but we don't know the order in which the re-tweets occurred as we don't have any information regarding the spread. Being able to analyse the next active node not only provides us some useful information about the graph, but it also allows us to predict the future influence and impact of the spread.

## Algorithms implemented for maximisation

### 1. Linear Threshold model
Initially we have a set of active nodes A. Also b<sub>v,w</sub> is an influence weight on a node v from its neighbor such that b<sub>v,w</sub> >=0 and b<sub>v,w</sub> <=1. Further in this model every node v is assigned a randomly chosen threshold θ<sub>v</sub> from [0,1]. The diffusion process then follows in discrete steps such that the nodes active in step t-1 remain active in step t also further activation of the inactive nodes take place if:The sum of influence weights from active neighbors of the node exceeds the threshold i.e Σ b<sub>v,w</sub> >= θv.

### 2. Independent Cascade Model

Given an initial set of active nodes A, an active node v in step t gets a single chance to activate its neighbor w with a success probability of p<sub>v,w</sub> . If v is successful then w gets activated in the diffusion step t+1. If unsuccessful v cannot make an attempt to activate w in the subsequent rounds which is not the case in linear threshold model. In case w has a set of activated neighbors then their attempts to activate w are sequences in an arbitrary fashion. The process continues until no more activations are possible. When node v becomes active, it is given a single chance to activate each currently inactive neighbor w • Succeeds with a probability 𝑝(𝑣,𝑤) (system parameter)
• Independent of history
• This probability is generally a coin flip (𝑈 [0,1])
• If v succeeds, then w will become active in step t+1; but whether or not v succeeds, it cannot make any further attempts to activate w in subsequent rounds. • If w has multiple newly activated neighbors, their attempts are sequenced in an arbitrary order.

## Tools and Technology used:
Deep Learning
Python 3.6
Scikit-learn,Numpy,Pandas
PyTorch , Keras :We have used PyTorch for building our feed neural network model and TensorFlow for building sequence to sequence model.
NetworkX: NetworkX library has been used to generate synthetic graphs for analysis.

## Dataset Used:
#### Synthetic:
1. Chain graph
2. Grid graph
3. Barabasi Albert graph A graph of n nodes is grown by attaching new nodes each with m edges that are preferentially attached to existing nodes with high degree.
4. Watts-Strogatz graph

#### Real:
5. Facebook TV Show 
6. Twitter real time data (future scope)

ROC AUC score variation with the number of epochs for fb_tvshow, chain, grid and barabasi_albert graph can be seen below:

<img width="815" alt="image" src="https://user-images.githubusercontent.com/37900145/150008216-65f9e827-268d-4c7f-8f24-f5c248f41947.png">
      
