import cv2 as cv
import numpy as np


class HomographyEstimator:
    def __init__(self, image1, image2, min_match_count=4):
        self.image1 = image1
        self.image2 = image2
        
        self.min_match_count = min_match_count
        
        self.sift = cv.SIFT_create()
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv.FlannBasedMatcher(index_params, search_params)
        
    def detect_keypoints_and_descriptors(self, image):
        """
        Detect keypoints and compute descriptors using SIFT.
        """
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray_image, None)
        
        return keypoints, descriptors
        
    def match_keypoints(self, descriptors1, descriptors2):
        """
        Match descriptors using the FLANN matcher.
        
        FLANN stands for Fast Library for Approximate Nearest Neighbors.
        It contains a collection of algorithms optimized for fast nearest
        neighbor search in large datasets and for high dimensional features.
        """
        matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
                
        return good_matches
    
    def build_homography(self):
        """
        Build the homography matrix using the matched keypoints.
        """
        homography = matched_image = None
        
        keypoints1, descriptors1 = self.detect_keypoints_and_descriptors(
            self.image1)
        keypoints2, descriptors2 = self.detect_keypoints_and_descriptors(
            self.image2)
        
        matches = self.match_keypoints(descriptors1, descriptors2)
        
        if len(matches) > self.min_match_count:
            image1_points = np.float32(
                [keypoints1[m.queryIdx].pt for m in matches]
            ).reshape(-1, 1, 2)
            
            image2_points = np.float32(
                [keypoints2[m.trainIdx].pt for m in matches]
            ).reshape(-1, 1, 2)
            
            homography, mask = cv.findHomography(
                image1_points,
                image2_points,
                cv.RANSAC,
                5.0
            )
            homography = homography.astype(np.float32)
            
            draw_params = dict(
                singlePointColor=None,
                matchesMask=mask.ravel().tolist(),
                flags=2
            )
            matched_image = cv.drawMatches(
                self.image1, keypoints1,
                self.image2, keypoints2,
                matches, None, **draw_params
            )
            
        return homography, matched_image
