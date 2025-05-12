import cv2 as cv
import numpy as np

from src.homography_estimator import HomographyEstimator


class Warping:
    def __init__(self, image_paths_list, center_idx):
        self.images = [cv.imread(image) for image in image_paths_list]
        self.center = center_idx

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
        defined by the homographies and applying blending.
        """
        panorama = None
        
        homographies_list = self.compute_cumulative_homographies()
        corners = []
        
        for image, homography in zip(self.images, homographies_list):
            h, w, _ = image.shape
            image_corners = np.float32(
                [[0, 0], [0, h], [w, h], [w, 0]]
            ).reshape(-1, 1, 2)
            transformed_corners = cv.perspectiveTransform(image_corners, homography)
            corners.append(transformed_corners)
        corners = np.vstack(corners)  # Use vstack to combine all corner arrays

        x_min, y_min = np.int32(corners.min(axis=0).ravel() - 0.5)
        x_max, y_max = np.int32(corners.max(axis=0).ravel() + 0.5)
        canvas_w, canvas_h = x_max - x_min, y_max - y_min
        
        panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        weight_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        
        translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], np.float32)
        
        for image, homography in zip(self.images, homographies_list):
            warped = cv.warpPerspective(
                image, translation_matrix @ homography, (canvas_w, canvas_h),
                flags=cv.INTER_LINEAR
            )
            
            # Create a weight map for blending
            mask = (warped > 0).any(axis=2).astype(np.float32)
            distance_transform = cv.distanceTransform(mask.astype(np.uint8), cv.DIST_L2, 5)
            normalized_weights = distance_transform / distance_transform.max()
            
            # Blend the warped image into the panorama
            for c in range(3):  # Iterate over color channels
                panorama[:, :, c] += warped[:, :, c] * normalized_weights
            weight_map += normalized_weights

        # Normalize the panorama by the weight map to avoid overexposure
        for c in range(3):
            panorama[:, :, c] /= np.maximum(weight_map, 1e-6)

        # Convert back to uint8 for display
        panorama = np.clip(panorama, 0, 255).astype(np.uint8)
        
        return panorama
