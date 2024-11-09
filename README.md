# SAM Segmentation Tool

This tool uses the SAM segmentation model to perform object segmentation on images.

## Setup

### Prerequisites
Developed in Python 3.9.

### Installation
1. Clone this repository.
2. Install dependencies using:
   ```bash
   pip install -r requirements.txt

## Usage
Run the script with:

```commandline
python segment_image.py --image_path path/to/image.png
```

### Arguments
- `--image_path`: Path to the input image (required).
- `--model_path`: Path to the SAM model checkpoint (default: models/sam_vit_h_4b8939.pth).
- `--output_dir`: Directory to save the output (defaults to input image directory).
- `--transparent_bg`: Save with a transparent background.
- `--override_save`: If False, prevents overwriting existing files by appending an index.

