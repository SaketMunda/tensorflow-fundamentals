# TensorFlow Fundamentals

This repository will help you to grow basic knowledge about [Google's TensorFlow](https://www.tensorflow.org/).

# What is TensorFlow and it's fundamentals

> What

TensorFlow is an open source end-to-end machine learning library for preprocessing data, modelling data and serving models (getting them into hands of others)

> Why

Rather than building machine learning models from scratch and train which requires huge computation power, separate GPUs/TPUs, it's more likely you'll use a library such as TensorFlow. It contains many of the most common machine learning functions you'll want to use.

> What is TPU & GPU ?

**TPU**, Tensor Processing Unit is an AI accelerator application-specific integrated circuit developed by Google for neural network machine learning, using Google's own TensorFlow software.

![image](https://user-images.githubusercontent.com/29537875/195336643-81229c04-a774-4a18-8c88-54291f6ca3e6.png)

**GPU**, Graphical Processing Unit, usually we use it for our personal computers to enhance the graphical interface of our CPUs and support high end softwares which requires high graphics.

![image](https://user-images.githubusercontent.com/29537875/195337734-82537f2d-ecce-4e38-be97-4ae07039f647.png)

# Topics covered in this repository

## 00. [Fundamentals of TensorFlow](https://github.com/SaketMunda/tensorflow-fundamentals/blob/master/00_tensorflow_fundamentals.ipynb)

### Highlights of the Section

* Introduction to tensors (How to create tensors)
* Getting information from tensors (tensor attributes/properties)
* Manipulating tensors (tensor operations)
* Using @tf.function (a way to speed up your regular Python functions)
* Using GPUs with TensorFlow
* Exercises to Try

### Notebook/Practicals

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaketMunda/tensorflow-fundamentals/blob/master/00_tensorflow_fundamentals.ipynb)

### Exercises

- [x] Create a vector, scalar, matrix and tensor with values of your choosing using tf.constant().
- [x] Find the shape, rank and size of the tensors you created in 1.
- [x] Create two tensors containing random values between 0 and 1 with shape [5, 300].
- [x] Multiply the two tensors you created in 3 using matrix multiplication.
- [x] Multiply the two tensors you created in 3 using dot product.
- [x] Create a tensor with random values between 0 and 1 with shape [224, 224, 3].
- [x] Find the min and max values of the tensor you created in 6 along the first axis.
- [x] Created a tensor with random values of shape [1, 224, 224, 3] then squeeze it to change the shape to [224, 224, 3].
- [x] Create a tensor with shape [10] using your own choice of values, then find the index which has the maximum value.
- [x] One-hot encode the tensor you created in 9.

### Extra-Curriculam

- [ ] Read through the list of [TensorFlow Python APIs](https://www.tensorflow.org/api_docs/python/), pick one we haven't gone through in this notebook, reverse engineer it (write out the documentation code for yourself) and figure out what it does.
- [ ] Try to create a series of tensor functions to calculate your most recent grocery bill (it's okay if you don't use the names of the items, just the price in numerical form).
  - How would you calculate your grocery bill for the month and for the year using tensors?
- [ ] Go through the [TensorFlow 2.x quick start for beginners](https://www.tensorflow.org/tutorials/quickstart/beginner) tutorial (be sure to type out all of the code yourself, even if you don't understand it).
  - Are there any functions we used in here that match what's used in there? Which are the same? Which haven't you seen before?
- [ ] Watch the video ["What's a tensor?"](https://www.youtube.com/watch?v=f5liqUk0ZTw) - a great visual introduction to many of the concepts we've covered in this notebook.

## 01. [Neural Network Regression with TensorFlow](https://github.com/SaketMunda/tensorflow-fundamentals/blob/master/01_neural_network_regression_in_tensorflow.ipynb) 

> What

There are many definition for a regression problem but in our case, we're going to simplify it to be: predicting a number.

For example,
* Predicting the selling price of houses given information about them (such as number of rooms, size, number of bathrooms)
* Predict the coordinates of a bounding box of an item in an image in an Object Detection problem
* Predict the cost of medical insurance for an individual given their demographics(age, sex, gender, race)

### Highlights of this section

* Architecture of a regression model
* Input shapes and Output shapes
  - `X`: features/data(inputs)
  - `y`: labels(outputs)
* Creating custom data to view and fit
* Steps in modelling
  - Creating a model
  - Compiling a model
    - Definining a loss function
    - Setting up an optimizer
    - Creating evaluation metrics
  - Fitting a model (getting it to find patterns in our data)
* Evaluating a model
  - Visualising the model ("visualise, visualise, visualise")
  - Looking at training curves
  - Compare predictions to ground truth (using our evaluation metrics)
* Saving a model
* Loading a model

### Notebook/Practicals

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaketMunda/tensorflow-fundamentals/blob/master/01_neural_network_regression_in_tensorflow.ipynb)

### Exercises

- [x] Create your own regression dataset (or make the one we created in "Create data to view and fit" bigger) and build fit a model to it.
- [x] Try building a neural network with 4 Dense layers and fitting it to your own regression dataset, how does it perform?
- [x] Try and improve the results we got on the insurance dataset, some things you might want to try include:
  * Building a larger model (how does one with 4 dense layers go?).
  * Increasing the number of units in each layer.
  * Lookup the documentation of [Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) and find out what the first parameter is, what happens if you increase it by 10x?
  * What happens if you train for longer (say 300 epochs instead of 200)?
- [x] Import the [Boston pricing dataset](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/boston_housing/load_data) from TensorFlow [tf.keras.datasets](https://www.tensorflow.org/api_docs/python/tf/keras/datasets) and model it.

### Extra-curriculum

- [x] [MIT introduction deep learning lecture 1](https://www.youtube.com/watch?v=7sB052Pz0sQ&ab_channel=AlexanderAmini) - gives a great overview of what's happening behind all of the code we're running.
- [ ] Reading: 1-hour of [Chapter 1 of Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html) by Michael Nielson - a great in-depth and hands-on example of the intuition behind neural networks.
- [ ] To practice your regression modelling with TensorFlow, I'd also encourage you to look through [Lion Bridge's collection of datasets](https://lionbridge.ai/datasets/) or [Kaggle's datasets](https://www.kaggle.com/data), find a regression dataset which sparks your interest and try to model.


## 02. [Neural Network Classification with TensorFlow](https://github.com/SaketMunda/tensorflow-fundamentals/blob/master/02_neural_network_classfication_with_tensorflow.ipynb)

> What 

A classification problem involves predicting whether something is one thing or another.

For example, you might want to:
* Predict whether or not someone has a heart disease based on their health parameters. This is called a **Binary classification**, since there are only two options.
* Predict whether a photo is of a food, a person or a dog. This is called **Multi-class classification** since there are more than two options.
* Predict what categories should be assigned to a wikipedia article. This is called **Multi-label classification** since a single article could have more than one category assigned.

### Highlights of this section

* Architecture of a classification model
* Input shapes and Output shapes
  - `X`: features/data(inputs)
  - `y`: labels(outputs)
* Creating custom data to view and fit
* Steps in modelling for binary and multi-class classification
  - Creating a model
  - Compiling a model
    - Definining a loss function
    - Setting up an optimizer
    - Creating evaluation metrics
  - Fitting a model (getting it to find patterns in our data)
  - Improving a model
* The power of non-linearity
* Evaluating a classification model
  - Visualising the model ("visualise, visualise, visualise")
  - Looking at training curves
  - Compare predictions to ground truth (using our evaluation metrics)
* Saving a model
* Loading a model

### Notebook/Practicals

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaketMunda/tensorflow-fundamentals/blob/master/02_neural_network_classfication_with_tensorflow.ipynb)

### Exercises

- [x] Play with neural networks in the [TensorFlow Playground](https://playground.tensorflow.org) for 10-minutes. Especially try different values of the learning, what happens when you decrease it? What happens when you increase it?
- [x] Replicate the model pictured in the [TensorFlow Playground diagram](https://playground.tensorflow.org/#activation=relu&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.001&regularizationRate=0&noise=0&networkShape=6,6,6,6,6&seed=0.51287&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false&regularization_hide=true&discretize_hide=true&regularizationRate_hide=true&percTrainData_hide=true&dataset_hide=true&problem_hide=true&noise_hide=true&batchSize_hide=true) below using TensorFlow code. Compile it using the Adam optimizer, binary crossentropy loss and accuracy metric. Once it's compiled check a summary of the model.
![Image](https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/02-tensorflow-playground-replication-exercise.png)
- [ ] Create a classification dataset using Scikit-Learn's [make_moons()](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) function, visualize it and then build a model to fit it at over 85% accuracy.
- [ ] Train a model to get 88%+ accuracy on the fashion MNIST test set. Plot a confusion matrix to see the results after.
- [ ] Recreate [TensorFlow's](https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax) [softmax activation](https://en.wikipedia.org/wiki/Softmax_function) function in your own code. Make sure it can accept a tensor and return that tensor after having the softmax function applied to it.
- [ ] Create a function (or write code) to visualize multiple image predictions for the fashion MNIST at the same time. Plot at least three different images and their prediction labels at the same time. Hint: see the [classification tutorial in the TensorFlow documentation](https://www.tensorflow.org/tutorials/keras/classification) for ideas.
- [ ] Make a function to show an image of a certain class of the fashion MNIST dataset and make a prediction on it. For example, plot 3 images of the `T-shirt` class with their predictions.

### Extra-curriculum

- [ ] Watch 3Blue1Brown's neural networks video 2: [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w). After you're done, write 100 words about what you've learned.
  - If you haven't already, watch video 1: [But what is a Neural Network?](https://www.youtube.com/watch?v=7sB052Pz0sQ&ab_channel=AlexanderAmini). Note the activation function they talk about at the end.
- [x] [MIT introduction deep learning lecture 1](https://www.youtube.com/watch?v=7sB052Pz0sQ&ab_channel=AlexanderAmini) - gives a great overview of what's happening behind all of the code we're running.
- [ ] Reading: 1-hour of [Chapter 1 of Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html) by Michael Nielson - a great in-depth and hands-on example of the intuition behind neural networks.
- [ ] Read the [ML-Glossary documentation on activation functions](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html). Which one is your favourite?
  - After you've read the ML-Glossary, see which activation functions are available in TensorFlow by searching "tensorflow activation functions".

# Resources

* [TensorFlow Deep Learning by Daniel Bourke](https://dev.mrdbourke.com/tensorflow-deep-learning)

