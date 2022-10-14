# TensorFlow Fundamentals

This repository will help you to grow basic knowledge about [Google's TensorFlow](https://www.tensorflow.org/).

## What is TensorFlow and it's fundamentals

> What

TensorFlow is an open source end-to-end machine learning library for preprocessing data, modelling data and serving models (getting them into hands of others)

> Why

Rather than building machine learning models from scratch and train which requires huge computation power, separate GPUs/TPUs, it's more likely you'll use a library such as TensorFlow. It contains many of the most common machine learning functions you'll want to use.

> Topics covered in this repository

* Introduction to tensors (How to create tensors)
* Getting information from tensors (tensor attributes/properties)
* Manipulating tensors (tensor operations)
* Using @tf.function (a way to speed up your regular Python functions)
* Using GPUs with TensorFlow
* Exercises to Try

> What is TPU & GPU ?

**TPU**, Tensor Processing Unit is an AI accelerator application-specific integrated circuit developed by Google for neural network machine learning, using Google's own TensorFlow software.

![image](https://user-images.githubusercontent.com/29537875/195336643-81229c04-a774-4a18-8c88-54291f6ca3e6.png)

**GPU**, Graphical Processing Unit, usually we use it for our personal computers to enhance the graphical interface of our CPUs and support high end softwares which requires high graphics.

![image](https://user-images.githubusercontent.com/29537875/195337734-82537f2d-ecce-4e38-be97-4ae07039f647.png)

## Fundamentals of Tensors with Examples

[Inside the Notebook](https://github.com/SaketMunda/tensorflow-fundamentals/blob/master/00_tensorflow_fundamentals.ipynb)

## TensorFlow Fundamentals Exercies

1. Create a vector, scalar, matrix and tensor with values of your choosing using tf.constant().
2. Find the shape, rank and size of the tensors you created in 1.
3. Create two tensors containing random values between 0 and 1 with shape [5, 300].
4. Multiply the two tensors you created in 3 using matrix multiplication.
5. Multiply the two tensors you created in 3 using dot product.
6. Create a tensor with random values between 0 and 1 with shape [224, 224, 3].
7. Find the min and max values of the tensor you created in 6 along the first axis.
8. Created a tensor with random values of shape [1, 224, 224, 3] then squeeze it to change the shape to [224, 224, 3].
9. Create a tensor with shape [10] using your own choice of values, then find the index which has the maximum value.
10. One-hot encode the tensor you created in 9.

## TensorFlow Fundamentals Extra-Curriculam

* Read through the list of [TensorFlow Python APIs](https://www.tensorflow.org/api_docs/python/), pick one we haven't gone through in this notebook, reverse engineer it (write out the documentation code for yourself) and figure out what it does.
* Try to create a series of tensor functions to calculate your most recent grocery bill (it's okay if you don't use the names of the items, just the price in numerical form).
  - How would you calculate your grocery bill for the month and for the year using tensors?
* Go through the [TensorFlow 2.x quick start for beginners](https://www.tensorflow.org/tutorials/quickstart/beginner) tutorial (be sure to type out all of the code yourself, even if you don't understand it).
  - Are there any functions we used in here that match what's used in there? Which are the same? Which haven't you seen before?
* Watch the video ["What's a tensor?"](https://www.youtube.com/watch?v=f5liqUk0ZTw) - a great visual introduction to many of the concepts we've covered in this notebook.

## Resources

* [TensorFlow Deep Learning by Daniel Bourke](https://dev.mrdbourke.com/tensorflow-deep-learning)

