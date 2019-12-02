'''
Script with Pytorch's dataloader class
'''

import os
import glob
import torch
import torch.utils.data as data
import torchvision

from typing import Tuple, List
from PIL import Image


class ImageLoader(data.Dataset):
  '''
  Class for data loading
  '''

  train_folder = 'train'
  test_folder = 'test'

  def __init__(self,
               root_dir: str,
               split: str = 'train',
               transform: torchvision.transforms.Compose = None):
    '''
    Init function for the class

    Args:
    - root_dir: the dir path which contains the train and test folder
    - split: 'test' or 'train' split
    - transforms: the transforms to be applied to the data
    '''
    self.root = os.path.expanduser(root_dir)
    self.transform = transform
    self.split = split

    if split == 'train':
      self.curr_folder = os.path.join(root_dir, self.train_folder)
    elif split == 'test':
      self.curr_folder = os.path.join(root_dir, self.test_folder)

    self.class_dict = self.get_classes()
    self.dataset = self.load_imagepaths_with_labels(self.class_dict)

  def load_imagepaths_with_labels(self, class_labels) -> List[Tuple[str, int]]:
    '''
    Fetches all image paths along with labels

    Args:
    -   class_labels: the class labels dictionary, with keys being the classes
        in this dataset
    Returns:
    -   list[(filepath, int)]: a list of filepaths and their class indices
    '''

    img_paths = []  # a list of (filename, class index)

    ###########################################################################
    # Student code begin
    ###########################################################################
    #path = "data" + "/**/*.jpg"
    #print(class_labels)
    if (self.curr_folder == "data/train"):
      self.curr_folder = "../" + self.curr_folder
    
    path = self.curr_folder + "/**/*.jpg"
    
    for file_path in glob.glob(path, recursive=True):
      img_class = file_path.split("/")[3]
      class_label = class_labels[img_class]
      img_paths.append((file_path, class_label))
      

    ###########################################################################
    # Student code end
    ###########################################################################
    return img_paths

  def get_classes(self) -> dict:
    '''
    Get the classes (which are folder names in self.curr_folder)

    Returns:
    -   Dict of class names (string) to integer labels
    '''

    classes = dict()
    ###########################################################################
    # Student code begin
    ###########################################################################
    names = []
    for folder in os.listdir(self.curr_folder):
      names.append(folder)
    
    classes['Bedroom'] = 11
    classes['Coast'] = 12
    classes['Forest'] = 7
    classes['Highway'] = 14
    classes['Industrial'] = 1
    classes['InsideCity'] = 3
    classes['Kitchen'] = 4
    classes['LivingRoom'] = 9
    classes['Mountain'] = 6
    classes['Office'] = 2
    classes['OpenCountry'] = 0
    classes['Store'] = 8
    classes['Street'] = 10
    classes[ 'Suburb'] = 13
    classes[ 'TallBuilding'] = 5


    ###########################################################################
    # Student code end
    ###########################################################################
    return classes

  def load_img_from_path(self, path: str) -> Image:
    '''
    Loads the image as grayscale (using Pillow)

    Note: do not normalize the image to [0,1]

    Args:
    -   path: the path of the image
    Returns:
    -   image: grayscale image loaded using pillow (Use 'L' flag while converting using Pillow's function)
    '''

    img = None

    ###########################################################################
    # Student code begin
    ###########################################################################
    #print(path)
    img = Image.open(path)
    img = img.convert('L')
    
    

    ###########################################################################
    # Student code end
    ###########################################################################
    return img

  def __getitem__(self, index: int) -> Tuple[torch.tensor, int]:
    '''
    Fetches the item (image, label) at a given index

    Note: Do not forget to apply the transforms, if they exist

    Hint:
    1) get info from self.dataset
    2) use load_img_from_path
    3) apply transforms if valid

    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    '''

    img = None
    class_idx = None

    ###########################################################################
    # Student code start
    ############################################################################
    img =  self.load_img_from_path(self.dataset[index][0])
    class_idx = self.dataset[index][1]
    img = self.transform(img)

    ############################################################################
    # Student code end
    ############################################################################
    return img, class_idx

  def __len__(self) -> int:
    """
    Returns the number of items in the dataset

    Returns:
        int: length of the dataset
    """

    l = 0

    ############################################################################
    # Student code start
    ############################################################################

    if (self.curr_folder == "data/train"):
      self.curr_folder = "../" + self.curr_folder
    
    path = self.curr_folder + "/**/*.jpg"
    
    for file_path in glob.glob(path, recursive=True):
      l+=1
    #print(l)

    ############################################################################
    # Student code end
    ############################################################################
    return l
