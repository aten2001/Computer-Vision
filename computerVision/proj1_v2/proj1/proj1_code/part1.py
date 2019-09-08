#!/usr/bin/python3

import numpy as np
import math


def create_Gaussian_kernel(cutoff_frequency):
  """
  Returns a 2D Gaussian kernel using the specified filter size standard
  deviation and cutoff frequency.

  The kernel should have:
  - shape (k, k) where k = cutoff_frequency * 4 + 1
  - mean = floor(k / 2)
  - standard deviation = cutoff_frequency
  - values that sum to 1

  Args:
  - cutoff_frequency: an int controlling how much low frequency to leave in
    the image.
  Returns:
  - kernel: numpy nd-array of shape (k, k)

  HINT:
  - The 2D Gaussian kernel here can be calculated as the outer product of two
    vectors with values populated from evaluating the 1D Gaussian PDF at each
    corrdinate.
  """

  ############################
  ### TODO: YOUR CODE HERE ###
  k = (cutoff_frequency *4) + 1
  mean = math.floor(k / 2)
  stdev = cutoff_frequency
  kern1dim = []
  for i in range(int(k)):
    prob =  (np.exp(-( (i-mean)**2 / (2*stdev**2) ) ) / np.sqrt(2*np.pi*stdev))
    kern1dim.append(prob)
  
  kernel = np.outer(kern1dim, kern1dim)
  kernel = (1/ np.sum(kernel)) * kernel

  ### END OF STUDENT CODE ####
  ############################

  return kernel

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of shape (m, n, c)
  - filter: numpy nd-array of shape (k, j)
  Returns
  - filtered_image: numpy nd-array of shape (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using OpenCV or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  """
  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###
  num_channels = image.shape[2]

  height_filter = filter.shape[0]
  width_filter = filter.shape[1]

  padded = np.pad(image, ((height_filter // 2 , ),(width_filter // 2,),(0,)), 'constant', constant_values=0 )
  filtered_img = np.zeros((image.shape[0], image.shape[1], num_channels))

  for channel in range(num_channels):
    for i in range(image.shape[0]):
      for j in range(image.shape[1]):
        kernel = padded[i:i+filter.shape[0], j:j+filter.shape[1],channel]
        filtered_img[i, j, channel] = np.sum((np.multiply(kernel, filter)), axis=(0,1))
  
  return filtered_img


  ### END OF STUDENT CODE ####
  ############################

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and a low-pass filter and creates a hybrid image. Returns
  the low frequency content of image1, the high frequency content of image 2,
  and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (x, y)
  Returns
  - low_frequencies: numpy nd-array of shape (m, n, c)
  - high_frequencies: numpy nd-array of shape (m, n, c)
  - hybrid_image: numpy nd-array of shape (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values of the hybrid image are between
    0 and 1. This is known as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]
  assert filter.shape[0] <= image1.shape[0]
  assert filter.shape[1] <= image1.shape[1]
  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###


  low_frequencies = my_imfilter(image1, filter)
  high_frequencies = image2 - my_imfilter(image2, filter)
  hybrid_image =  np.add(high_frequencies, low_frequencies)
  hybrid_image = np.clip(hybrid_image, 0 , 1)
  

  ### END OF STUDENT CODE ####
  ############################

  return low_frequencies, high_frequencies, hybrid_image
