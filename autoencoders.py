

"""##Importing the libraries"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

"""## Importing the dataset"""

# We won't be using this dataset.
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

"""## Preparing the training set and the test set"""

training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

"""## Getting the number of users and movies"""

nb_users = int(max(max(training_set[:, 0], ), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1], ), max(test_set[:, 1])))

"""## Converting the data into an array with users in lines and movies in columns"""

def convert(data):
  new_data = []
  for id_users in range(1, nb_users + 1):
    id_movies = data[:, 1] [data[:, 0] == id_users]
    id_ratings = data[:, 2] [data[:, 0] == id_users]
    ratings = np.zeros(nb_movies)
    ratings[id_movies - 1] = id_ratings
    new_data.append(list(ratings))
  return new_data
training_set = convert(training_set)
test_set = convert(test_set)

"""## Converting the data into Torch tensors"""

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

"""## Creating the architecture of the Neural Network"""
#inheriting the parent class from torch
#inheritance is used so that we can use methods of parent class.
#we are using stacked auto encoders as there are a number of hidden layers based on the input.
class SAE(nn.Module):                        #inheritance
    def __init__(self, ):                    #, is added so the the function will consider the variable of parent class
        super(SAE, self).__init__()          #for inheriting methods of parent class
        self.fc1 = nn.Linear(nb_movies, 20)  #connecting input layer to first hidden layer,20 is number of nodes in hidden layer
        self.fc2 = nn.Linear(20, 10)         #connecting 1st hidden layer to 2nd ideen layer containing 10 neurons or nodes
        self.fc3 = nn.Linear(10, 20)         #3rd hidden layer connection while maintaining symmetry
        self.fc4 = nn.Linear(20, nb_movies)  #the o/p is reconstruction of i/p layer 
        self.activation = nn.Sigmoid()
        
    def forward(self, x):                 #x is the i/p vector of features
        #right x is i/p vector of features on which encoding is done and the left x is new encoded vector.
        x = self.activation(self.fc1(x))  #encoding the first hidden layer
        x = self.activation(self.fc2(x))  #same
        x = self.activation(self.fc3(x))  #same
        x = self.fc4(x)                   #since it is the last layer i.e the o/p layer encoding is not reqd. on it
        return x                          #now x is the vector of predicted ratings
    
sae = SAE()
criterion = nn.MSELoss() #object of MSE class from nn module
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)
#RMSprop is the optmizer class from optim module
#sae.parameters() indicates all the parameters and methods of SAE class
#lr is learning rate and weight_decay is used to reduce learning rate after each epoch to regulaize

"""## Training the SAE"""

nb_epoch = 200    
for epoch in range(1, nb_epoch + 1):
  train_loss = 0
  s = 0.
  for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = input.clone()                #input vector is clones or copied in target
    if torch.sum(target.data > 0) > 0:    #target.data is all the ratings of a user
      output = sae(input)                 #goes to the forward fuction and returns the predicted ratings
      target.require_grad =  False        #for optmiazation of code
      output[target == 0] = 0             #if ratings of movie is 0 after prediction than it will not participate in computatiuon of erroe 
      loss = criterion(output, target)    #Computation of loss error
      mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
      loss.backward()
      train_loss =train_loss + np.sqrt(loss.data*mean_corrector)
      s =s + 1.
      optimizer.step()
  print('epoch: '+str(epoch)+'loss: '+ str(train_loss/s))

"""## Testing the SAE"""

test_loss = 0
s = 0.
for id_user in range(nb_users):
  input = Variable(training_set[id_user]).unsqueeze(0)
  target = Variable(test_set[id_user]).unsqueeze(0)
  if torch.sum(target.data > 0) > 0:
    output = sae(input)
    target.require_grad = False
    output[target == 0] = 0
    loss = criterion(output, target)
    mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
    test_loss =test_loss + np.sqrt(loss.data*mean_corrector) 
    s =s + 1.
print('test loss: '+str(test_loss/s))