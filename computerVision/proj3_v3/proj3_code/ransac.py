import numpy as np
import math
from proj3_code.least_squares_fundamental_matrix import solve_F
from proj3_code import two_view_data
from proj3_code import fundamental_matrix


def calculate_num_ransac_iterations(prob_success, sample_size, ind_prob_correct):
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int the number of samples included in each RANSAC iteration
    -   ind_prob_success: float the probability that each element in a sample is correct

    Returns:
    -   num_samples: int the number of RANSAC iterations needed

    """
    ##############################
    # TODO: Student code goes here

    # N = log(1-p) / (log(1-(1-e)^s))
    numerator = np.log2(1- prob_success)
    e = ind_prob_correct ** sample_size
    denom = np.log2(1-e)
    num_samples = numerator / denom
    ##############################

    return int(num_samples)


def find_inliers(x_0s, F, x_1s, threshold):
    """ Find the inliers' indices for a given model.

    There are multiple methods you could use for calculating the error
    to determine your inliers vs outliers at each pass. CoHowever, we suggest
    using the line to point distance function we wrote for the
    optimization in part 2.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   F: The proposed fundamental matrix
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    -   threshold: the maximum error for a point correspondence to be
                    considered an inlier
    Each row in x_1s and x_0s is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -    inliers: 1D array of the indices of the inliers in x_0s and x_1s

    """
    ##############################
    # TODO: Student code goes here
    inliers = []
    for idx in range(len(x_1s)):
        line = np.matmul(F, x_1s[idx])
        line_T = np.matmul(F.transpose(), x_0s[idx])
        dist = fundamental_matrix.point_line_distance(line, x_0s[idx])
        dist2 = fundamental_matrix.point_line_distance(line_T, x_1s[idx])
        if (dist + dist2 < 2* threshold): inliers.append(idx)

    ##############################

    return np.array(inliers)


def ransac_fundamental_matrix(x_0s, x_1s):
    """Find the fundamental matrix with RANSAC.

    Use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You will call your
    solve_F() from part 2 of this assignment
    and calculate_num_ransac_iterations().

    You will also need to define a new function (see above) for finding
    inliers after you have calculated F for a given sample.

    Tips:
        0. You will need to determine your P, k, and p values.
            What is an acceptable rate of success? How many points
            do you want to sample? What is your estimate of the correspondence
            accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for
            creating your random samples
        2. You will want to call your function for solving F with the random
            sample and then you will want to call your function for finding
            the inliers.
        3. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 1.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    Each row is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_x_0: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the left image that are inliers with
                   respect to best_F
    -   inliers_x_1: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the right image that are inliers with
                   respect to best_F

    """
    ##############################
    # TODO: Student code goes here
    """best_F = np.zeros((3,3))
    max_inliers = 0
    inliers_x_0 = np.zeros((3,3))
    inliers_x_1 = np.zeros((3,3))
    num_iterations = calculate_num_ransac_iterations(.9, 9, 0.7)

    for i in range(num_iterations):
        random_x_idxs = np.random.choice(x_0s.shape[0], 9)
        # random_x1s_idxs = np.random.choice(x_1s.shape[0], 9)
        random_x0s = np.array([x_0s[i] for i in random_x_idxs])
        random_x1s = np.array([x_1s[i] for i in random_x_idxs])
        curr_fundamental_matrix = solve_F(random_x0s, random_x1s)
        homo_x0s, homo_x1s = two_view_data.preprocess_data(x_0s, x_1s)
        curr_inliers = find_inliers(homo_x0s, curr_fundamental_matrix, homo_x1s, 1)
        if (curr_inliers.shape[0] > max_inliers):
            max_inliers = curr_inliers.shape[0]
            best_F = curr_fundamental_matrix
            inliers_x_0 = np.array([x_0s[i] for i in curr_inliers])
            inliers_x_1 = np.array([x_1s[i] for i in curr_inliers])"""
    best_model = np.zeros((3,3))
    inliers_x_0 = best_model.copy()
    inliers_x_1 = best_model.copy()
    iterations = calculate_num_ransac_iterations(0.9, 9, 0.7)
    num_inliers = 0
    for iteration in range(iterations):
        length = x_0s.shape[0]
        possible_inliers = np.random.choice(length, 9)
        p_x1s = []
        p_x0s = []
        for p_inlier in possible_inliers:
            p_x1s[p_inlier] = x_1s[p_inlier]
            p_x0s[p_inlier] = x_0s[p_inlier]
        test_inliers = find_inliers(
             two_view_data.preprocess_data(x_0s, x_1s)[0],
             solve_F(p_x0s, p_x1s),
             two_view_data.preprocess_data(x_0s, x_1s)[1],
             1)
        count_curr_inliers = test_inliers.shape[0]
        if (num_inliers < count_curr_inliers):
            for idx in test_inliers:
                inliers_x_0[idx] = x_1s[idx]
                inliers_x_1[idx] = x_0s[idx]
            best_model = solve_F(p_x0s, p_x1s)
            num_inliers = test_inliers.shape[0]
    best_F = best_model
    inliers_x_1 = np.array(inliers_x_1)
    inliers_x_0 = np.array(inliers_x_0)

    ##############################

    return best_F, inliers_x_0, inliers_x_1


def test_with_epipolar_lines():
    """Unit test you will create for your RANSAC implementation.

    It should take no arguments and it does not need to return anything,
    but it **must** display the images when run.

    Use the code in the jupyter notebook as an example for how to open the
    image files and perform the necessary operations on them in our workflow.
    Remember the steps are Harris, SIFT, match features, RANSAC fundamental matrix.

    Display the proposed correspondences, the true inlier correspondences
    found by RANSAC, and most importantly the epipolar lines in both of your images.
    It should be clear that the epipolar lines intersect where the second image
    was taken, and the true point correspondences should indeed be good matches.

    """
    ##############################
    # TODO: Student code goes here
    raise NotImplementedError
    ##############################
