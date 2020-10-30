# Deep Learning with fashion MNIST

After spending several weeks on studying the theory behind Deep Learning and Neural Network, I wanted to get hands-on experience and practice implementing NNs with PyTorch. The dataset I use for this project is [fashionMNIST](https://www.kaggle.com/zalando-research/fashionmnist), which has the same properties as the famous MNIST dataset, but uses pictures of ten types of Zalando articles instead of hand-written digits.

<p align="center">
  <img width="350" height="250" src="https://cdn.pixabay.com/photo/2016/03/31/23/37/blouse-1297721_960_720.png">
</p>

***

#### Project updates

[10-2020]
 - conducting experiments with various different CNN models 
 - testing the best model on the test set
 - evaluating the results
 
 [09-2020]
 - preprocessing the data
 - building a dataset and dataloader
 - building a baseline model (NN with two fully connected layers) 
 
***

#### The dataset
The images used for this project are 28x28x1 images of clothing items on the online selling platform Zalando. The dataset includes 70,000 images, split into a training and a test set. The images are labeled according to 10 classes: t-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag and ankle boot.

<p align="left">
  <img width="500" height="200" src="https://github.com/HeleneFabia/fashion-mnist/blob/master/images/data.png">
</p>

***

#### Training a model

After building a baseline model (a neural network with two fully connected layers), I experimented with different CNN architectures and hyperparameters, until I was satisfied with the result. The final model was the following:

<p align="left">
  <img width="490" height="150" src="https://github.com/HeleneFabia/fashion-mnist/blob/master/images/model.png">
</p>

Out of all my experiments, this model achieved the highest accuracy while also keeping the training and validation loss low without overfitting.

<p align="left">
  <img width="380" height="230" src="https://github.com/HeleneFabia/fashion-mnist/blob/master/images/learning_curve.png">
</p>

***

#### Evaluation 

The model achieved 92.27% accuracy on the set set. In order to look at the weaknesses of the model, I visualized a confusion matrix as well as the worst predictions the model made.

<p align="left">
  <img width="400" height="300" src="https://github.com/HeleneFabia/fashion-mnist/blob/master/images/confusion_matrix.png">
</p>

Here it can be seen that the model often confuses shirts with t-shirts, pullovers, coats, and sometimes also dresses.

<p align="left">
  <img width="720" height="550" src="https://github.com/HeleneFabia/fashion-mnist/blob/master/images/top_25_wrong_preds.png">
</p>

Furthermore, the worst 25 predictions (i.e. when the model output a wrong prediction with very high confidence) are quite difficult cases, for example, t-shirts that look like dresses or sneakers that look like ankle boots and vice versa.

***

#### Next steps and ideas for improvement

- using k-fold cross validation to select the best model
- Using data augmentation to increase the size of the training set
- Using a pre-trained model such as ResNet 

***

Have a look at [my notebook](https://github.com/HeleneFabia/fashion-mnist/blob/master/fashion-mnist-cnn.ipynb) for more details! 
