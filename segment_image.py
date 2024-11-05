import os
import argparse
import numpy as np
import cv2
from skimage import io, transform
from segment_anything import SamPredictor, sam_model_registry


def initialize_predictor(model_path):
    """Initialize SAM model predictor."""
    sam = sam_model_registry["vit_h"](checkpoint=model_path)
    return SamPredictor(sam)


def load_image(image_path, target_size=1024):
    """Load and resize image to a specified size while maintaining aspect ratio."""
    image = io.imread(image_path)
    height, width = image.shape[:2]
    scale_factor = target_size / max(height, width)
    image = transform.resize(image, (int(height * scale_factor), int(width * scale_factor)))
    return np.asarray(image * 255, dtype=np.uint8)[:, :, :3]


def create_mask(predictor, points, labels, image):
    """Generate and apply mask to image based on selected points and labels."""
    prompt_coords = np.array(points)
    prompt_labels = np.array(labels)
    masks, _, _ = predictor.predict(point_coords=prompt_coords, point_labels=prompt_labels)
    mask = (np.sum(masks, axis=0) > 0).astype(np.uint8) * 255
    return mask


def generate_unique_filename(output_dir, base_name, extension):
    """Generate a unique filename by appending an index if a file already exists."""
    counter = 1
    output_path = os.path.join(output_dir, f"{base_name}_segmentation{extension}")
    while os.path.exists(output_path):
        output_path = os.path.join(output_dir, f"{base_name}_segmentation_{counter}{extension}")
        counter += 1
    return output_path


def save_segmented_image(image, mask, output_dir, base_name, transparent_bg, override_save):
    """Save the segmented image with either a white or transparent background."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        outer_contour_mask = np.zeros_like(mask)
        cv2.drawContours(outer_contour_mask, contours, -1, 255, thickness=cv2.FILLED)

        segmented_image = cv2.bitwise_and(image, image, mask=outer_contour_mask)
        segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

        if transparent_bg:
            result = np.dstack((segmented_image_bgr, outer_contour_mask))  # Add alpha channel
        else:
            white_bg = np.ones_like(image) * 255
            white_bg_bgr = cv2.cvtColor(white_bg, cv2.COLOR_RGB2BGR)
            result = np.where(outer_contour_mask[..., None], segmented_image_bgr, white_bg_bgr)

        extension = ".png"
        if override_save:
            output_path = os.path.join(output_dir, f"{base_name}_segmentation{extension}")
        else:
            output_path = generate_unique_filename(output_dir, base_name, extension)

        cv2.imwrite(output_path, result)
        print(f"Segmented image saved at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Segment an object from an image using SAM model.")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--model_path", default="models/sam_vit_h_4b8939.pth", help="Path to the SAM model checkpoint file")
    parser.add_argument("--output_dir", help="Directory to save the segmented image (defaults to input image directory)")
    parser.add_argument("--transparent_bg", action="store_true", help="Save image with transparent background")
    parser.add_argument("--target_size", type=int, default=1024, help="Target size for image resizing (default: 1024)")
    parser.add_argument("--override_save", type=bool, default=True, help="Override save if file exists (default: True)")

    args = parser.parse_args()

    # Validate image path and determine output directory
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}")
        return

    # Set output directory to the input image's directory if not specified
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.image_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    predictor = initialize_predictor(args.model_path)
    image = load_image(args.image_path, args.target_size)
    predictor.set_image(image)

    points, labels = [], []
    masks = None

    def select_point(event, x, y, flags, param):
        nonlocal masks, points, labels
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            labels.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append([x, y])
            labels.append(0)
        if points:
            masks = create_mask(predictor, points, labels, image)
            display_image = image.copy()
            contours, _ = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display_image, contours, -1, (0, 255, 0), 2)
            cv2.imshow('Image', cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))
            cv2.imshow('Mask', masks)

    base_name = os.path.splitext(os.path.basename(args.image_path))[0]

    cv2.imshow('Image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.setMouseCallback('Image', select_point)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and masks is not None:
            save_segmented_image(image, masks, output_dir, base_name, args.transparent_bg, args.override_save)
            break
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
