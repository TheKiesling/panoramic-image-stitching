import cv2 as cv
import numpy as np

from src.homography_estimator import HomographyEstimator


class Warping:
    def __init__(self, image_paths_list, center_idx, blend=False):
        self.images = [cv.imread(image) for image in image_paths_list]
        self.center = center_idx
        self.blend = blend

    def compute_cumulative_homographies(self):
        """
        Compute the homographies to the center image for all images.
        The center image is the one at index self.center.
        The homographies are computed from left to right and then
        from right to left. The homography for the center image is
        the identity matrix.
        """
        homographies = []
        
        homographies_neighbor = []
        for i in range(len(self.images) - 1):
            H, _ = HomographyEstimator(
                self.images[i], self.images[i + 1]
            ).build_homography()
            homographies_neighbor.append(H.astype(np.float32))

        homographies = [None] * len(self.images)
        homographies[self.center] = np.eye(3, dtype=np.float32)

        for i in range(self.center - 1, -1, -1):
            homographies[i] = homographies_neighbor[i] @ homographies[i + 1]

        for i in range(self.center + 1, len(self.images)):
            homographies[i] = np.linalg.inv(homographies_neighbor[i - 1]) @ homographies[i - 1]

        return homographies

    def create_panorama(self):
        """
        Create the panorama by warping all images to the canvas
        defined by the homographies.
        """
        panorama = None
        
        homographies_list = self.compute_cumulative_homographies()
        corners = []
        
        for image, homography in zip(self.images, homographies_list):
            h, w, _ = image.shape
            corners_image = np.float32(
                [[0, 0], [0, h], [w, h], [w, 0]]
            ).reshape(-1, 1, 2)
            corners.append(cv.perspectiveTransform(corners_image, homography))
        corners = np.concatenate(corners, axis=0)

        x_min, y_min = np.int32(corners.min(axis=0).ravel() - 0.5)
        x_max, y_max = np.int32(corners.max(axis=0).ravel() + 0.5)
        canvas_width, canvas_height = x_max - x_min, y_max - y_min
        
        accumulated_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
        accumulated_weight = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        
        panorama = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], np.float32)
        
        for image, homography in zip(self.images, homographies_list):
            warped = cv.warpPerspective(
                image, translation_matrix @ homography, (canvas_width, canvas_height),
                flags=cv.INTER_LINEAR
            )
            mask = (warped > 0).any(axis=2)
            
            if not self.blend:
                panorama[mask] = warped[mask]
            else:
                mask = (warped > 0).any(axis=2).astype(np.uint8)
                dist = cv.distanceTransform(mask, cv.DIST_L2, 3)
                weight = dist / dist.max()

                for c in range(3):
                    accumulated_image[:, :, c] += warped[:, :, c].astype(np.float32) * weight
                accumulated_weight += weight
                
        if self.blend:
            panorama = (accumulated_image / (accumulated_weight[..., None] + 1e-8)).astype(np.uint8)
            
        return panorama
