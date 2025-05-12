import os
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from matplotlib.widgets import Slider, Button

from src.warping import Warping


def process_subfolders(data_dir):
    """
    Process each subfolder in the data directory and create a panorama for each.
    """
    subfolders = [
        subfolder for subfolder in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, subfolder))
    ]

    if not subfolders:
        print("No subfolders found in the data directory.")
        return

    cols = 2  # Number of columns in the grid
    rows = (len(subfolders) + cols - 1) // cols  # Calculate rows needed
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    sliders = []  # List to store sliders for each subplot
    warping_objects = []  # Store Warping objects for each subfolder
    img_displays = []  # Store image display objects for each subplot

    for ax, subfolder in zip(axes, subfolders):
        subfolder_path = os.path.join(data_dir, subfolder)
        images_path = [
            os.path.join(subfolder_path, img)
            for img in sorted(os.listdir(subfolder_path))
            if img.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if len(images_path) < 2:
            ax.axis("off")
            print(f"Skipping '{subfolder}' (not enough images).")
            continue

        print(f"Processing folder: {subfolder}")

        # Initial center index
        initial_center = len(images_path) // 2
        warping = Warping(images_path, initial_center)
        panorama = warping.create_panorama()

        # Display the initial panorama
        img_display = ax.imshow(cv.cvtColor(panorama, cv.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_title(subfolder)

        # Create a slider for this subplot
        slider_ax = plt.axes([ax.get_position().x0, ax.get_position().y0 - 0.05,
                              ax.get_position().width, 0.03])
        slider = Slider(slider_ax, "Center", 0, len(images_path) - 1,
                        valinit=initial_center, valstep=1)
        sliders.append(slider)
        warping_objects.append(warping)
        img_displays.append(img_display)

    import gc  # Importar el módulo de recolección de basura

    def refresh(event):
        """
        Refresh all panoramas based on the current slider values.
        """
        # Liberar memoria antes de recalcular
        for img_display in img_displays:
            img_display.set_data(np.zeros((1, 1, 3), dtype=np.uint8))  # Placeholder
        gc.collect()  # Forzar la recolección de basura

        # Recalcular los panoramas
        for slider, warping, img_display in zip(sliders, warping_objects, img_displays):
            center_idx = int(slider.val)
            if warping.center != center_idx:  # Solo recalcular si el centro cambió
                warping.center = center_idx
                panorama = warping.create_panorama()
                img_display.set_data(cv.cvtColor(panorama, cv.COLOR_BGR2RGB))
        fig.canvas.draw_idle()

    # Add a refresh button
    button_ax = plt.axes([0.4, 0.01, 0.2, 0.05])  # Position of the button
    button = Button(button_ax, "Refresh")
    button.on_clicked(refresh)

    # Hide any unused subplots
    for ax in axes[len(subfolders):]:
        ax.axis("off")

    plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust spacing between subplots
    plt.show()


if __name__ == "__main__":
    data_dir = "data"  # Path to the data directory
    process_subfolders(data_dir)