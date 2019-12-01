import glob
import os
import numpy as np

from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> (np.array, np.array):
  '''
  Compute the mean and the standard deviation of the dataset.

  Note: convert the image in grayscale and then in [0,1] before computing mean
  and standard deviation

  Hints: use StandardScalar (check import statement)

  Args:
  -   dir_name: the path of the root dir
  Returns:
  -   mean: mean value of the dataset (np.array containing a scalar value)
  -   std: standard deviation of th dataset (np.array containing a scalar value)
  '''

  mean = None
  std = None

  ############################################################################
  # Student code begin
  ############################################################################
  path = dir_name + "/**/*.jpg"
  scalar = StandardScaler()
  for file_path in glob.glob(path, recursive=True):
    img = Image.open(file_path)
    arr = np.array(img)
    #print(arr.shape)
    arr = np.array(img) / 255
    arr = arr.flatten()
    data = arr.reshape(arr.shape[0], 1)
    #print(data.shape)
    scalar.partial_fit(data)

  mean = scalar.mean_
  std = scalar.var_
  std[0] = np.sqrt(std[0])

  ############################################################################
  # Student code end
  ############################################################################
  return mean, std
