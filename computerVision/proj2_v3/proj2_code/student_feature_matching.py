import numpy as np
import torch


def compute_feature_distances(features1, features2):
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second set
      features (m not necessarily equal to n)

    Returns:
    - dists: A numpy array of shape (n,m) which holds the distances from each
      feature in features1 to each feature in features2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    x = features1
    y = features2
    n = x.shape[0]
    m = y.shape[0]
    dist_x = np.zeros((n,))
    dist_y = np.zeros((m,))

    """xT = np.matmul(np.transpose(x),x)
    print(xT.shape)
    yT = np.matmul(np.transpose(y), y)
    print(yT.shape)
    dists = np.zeros((x.shape[0], y.shape[0]))

    xT = np.divide(xT, x.shape[0])
    yT = np.divide(yT, y.shape[0])

    norm = np.linalg.norm(xT - yT)

    dists = np.tile(norm, (x.shape[0], y.shape[0]))"""

    print(x.shape)
    
    d0 = np.subtract.outer(x[:,0], y[:,0])
    d1 = np.subtract.outer(x[:,1], y[:,1])
    dists = np.hypot(d0,d1)
    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dists


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second
      set of features (m not necessarily equal to n)

    Returns:
    - matches: A numpy array of shape (k,2), where k is the number of matches.
      The first column is an index in features1, and the second column is an
      index in features2
    - confidences: A numpy array of shape (k,) with the real valued confidence
      for every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    distances = compute_feature_distances(features1, features2)
  
    m = []
    confidences = []
    for i in range(len(features1)):
      d = distances[i]
      fb1 = np.argsort(d)[0]
      fb2 = np.argsort(d)[1]
      d1 = d[fb1]
      d2 = d[fb2]
      score = (d1 / d2) 
      if score < 0.8:
        m.append((i, fb1))
        confidences.append(score)
    
    k = len(m)
    matches = np.zeros((k,2), dtype=int)
    
   
    for i in range(len(m)):
      matches[i][0] = m[i][0]
      matches[i][1] = m[i][1]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return matches, confidences
