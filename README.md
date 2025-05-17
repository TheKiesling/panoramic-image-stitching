# Panoramic Image Builder

A Python-based tool for creating panoramic images by stitching multiple overlapping photos together. This project uses OpenCV and computer vision techniques to align and blend images seamlessly.

## Features

- Automatic image alignment and warping
- Seamless blending of overlapping regions
- Support for multiple input images
- Customizable center image selection
- Interactive visualization using matplotlib

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

## Usage

1. Install the required dependencies:
```bash
pip install -r requirements.txt
``` 
2. Place your images in the `images/` directory
3. Modify the `images_path` list in `main.py` to include your image paths
4. Run the script:
```bash
python main.py
```

## Project Structure

```
panoramic-image-builder/
├── src/
│   ├── homography_estimator.py     # Homography matrix estimation for image alignment
│   └── warping.py                  # Image warping and blending implementation
├── images/                         # Directory for input images
├── main.py                         # Main script
├── requirements.txt                # Project dependencies
└── README.md
```
