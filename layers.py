import tensorflow as tf
from tensorflow.keras import layers

def Convolution(input, depth, kernel, strides=(2,2), activation=layers.LeakyReLU) :
  """
  Return a Convolutional layer followed by Batch normalisation and given activation
  input : previous model's layer
  depth : number of convolution filters
  kernel : convolution's kernel size
  stride : convolution's stride size, act as a downsize factor
  activation : final activation layer
  """
  tmp = layers.Conv2D(depth, kernel, strides=strides, padding='same')(input)
  tmp = layers.BatchNormalization()(tmp)
  return activation()(tmp)

def Deconvolution(input, depth, kernel, strides=(2,2), activation=layers.LeakyReLU) :
  """
  Return a Convolutional layer followed by Batch normalisation and given activation
  input : previous model's layer
  depth : number of convolution filters
  kernel : convolution's kernel size
  stride : convolution's stride size, act as a upsize factor
  activation : final activation layer
  """
  tmp = layers.Conv2DTranspose(depth, kernel, strides=strides, padding='same')(input)
  tmp = layers.BatchNormalization()(tmp)
  return activation()(tmp)

def DilatedConvolution(input, depth, kernel, dilatation=(2,2), activation=layers.LeakyReLU) :
  """
  Return a Dilated Convolutional layer followed by Batch normalisation and given activation
  input : previous model's layer
  depth : number of convolution filters
  kernel : convolution's kernel size
  dilatation : dilatation's factor
  activation : final activation layer
  """
  tmp = layers.Conv2D(depth, kernel, dilation_rate=dilatation, padding='same')(input)
  tmp = layers.BatchNormalization()(tmp)
  return activation()(tmp)