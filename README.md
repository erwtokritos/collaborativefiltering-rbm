# collaborativefiltering-rbm
This is a Java implementation of the paper [Restricted Boltzmann Machines for Collaborative Filtering](http://www.machinelearning.org/proceedings/icml2007/papers/407.pdf) by Salakhutdinov et al.  The authors propose a two-layer undirected graphical model that is suited for collaborative filtering tasks. More specifically, a different RBM is used for each user under the following constraints: 

1. Every RBM has the same number of hidden units, but an RBM has active softmax visible units only for the items rated by that user
2. The connection weights (and biases) between the softmax visible layer and the hidden layer are tied i.e. if two users have rated the same movie, their two RBM's must use the same weights between the softmax unit for that movie and the hidden units

The network is trained by optimizing the Contrastive Divergence (CD). The main class is the 'CollaborativeFilteringRBM' class which resides in the 'deeplearning\tools\rbmforcollaborativefiltering' package. It supports three methods:

* loadRatings(String file), which loads and transforms the rating data. It expects a tab separated file in the form : 'user'  'item'  'rating'. For convenience, I have included a snapshot of the Movielens dataset under the 'data' folder
* fit(RbmOptions) : It optimizes the parameters of the network for the given data
* predict(String userId, String itemId, PredictionType predictionType) : It predicts the rating for a (user, item) pair. PredictionType can be either MAX (get max prob choice) or MEAN (weighted mean of all ratings)


I have used this algorithm in my paper [Trust Inference in Online Social Networks](http://dl.acm.org/citation.cfm?id=2809418). If you find this contribution useful, I would appreciate it if you cited my paper :

Athanasios Papaoikonomou, Magdalini Kardara, and Theodora Varvarigou. 2015. Trust Inference in Online Social Networks. In Proceedings of the 2015 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining 2015 (ASONAM '15), Jian Pei, Fabrizio Silvestri, and Jie Tang (Eds.). ACM, New York, NY, USA, 600-604. DOI=http://dx.doi.org/10.1145/2808797.2809418
