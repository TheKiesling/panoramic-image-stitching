import matplotlib.pyplot as plt

from src.homography_estimator import HomographyEstimator


if __name__ == "__main__":
    estimator = HomographyEstimator('images/room1.jpg', 'images/room2.jpg')
    homography_matrix, matched_image = estimator.build_homography()

    if homography_matrix is not None:
        plt.imshow(matched_image, cmap='gray')
        plt.axis('off')
        plt.show()
