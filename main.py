import matplotlib.pyplot as plt
import cv2 as cv

from src.warping import Warping


if __name__ == "__main__":
    images_path = [
        "images/S1.jpg",
        "images/S2.jpg",
        "images/S3.jpg",
        "images/S5.jpg",
        "images/S6.jpg",
    ]
    image_center = 2
    warping = Warping(images_path, image_center)
    
    panorama = warping.create_panorama()
    
    plt.imshow(cv.cvtColor(panorama, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
