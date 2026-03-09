import os
import cv2
import numpy as np
import shutil
import random
from collections import defaultdict

base_folder = "YOUR BASE FOLDER"
images_dir = os.path.join(base_folder, "images")
labels_dir = os.path.join(base_folder, "labels")
class_organized_dir = os.path.join(base_folder, "organized_by_class")

# Classes that need augmentation (Just add your good and bad classes with their count here)
classes_to_augment = {
    "good": 50,
    "bad": 50
}

# How many images do you want to generate by augmentation? Just update this value
target_count = 1500

def rotate_image_and_bboxes(image, bbox_lines, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    new_bbox_lines = []
    
    for bbox_line in bbox_lines:
        parts = bbox_line.strip().split()
        if len(parts) != 5:
            continue
            
        class_id = parts[0]
        x_center, y_center, width, height = map(float, parts[1:])
        
        # Convert to absolute coordinates
        abs_x = x_center * w
        abs_y = y_center * h
        abs_w = width * w
        abs_h = height * h
        
        # Get corner points of bounding box
        x1 = abs_x - abs_w/2
        y1 = abs_y - abs_h/2
        x2 = abs_x + abs_w/2
        y2 = abs_y + abs_h/2
        
        # Apply rotation to bbox corners
        corners = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]])
        rotated_corners = corners @ M.T
        
        # Get new bounding box
        new_x1 = np.min(rotated_corners[:, 0])
        new_y1 = np.min(rotated_corners[:, 1])
        new_x2 = np.max(rotated_corners[:, 0])
        new_y2 = np.max(rotated_corners[:, 1])
        
        # Clamp to image boundaries
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(w, new_x2)
        new_y2 = min(h, new_y2)
        
        # Skip if bounding box becomes too small
        if (new_x2 - new_x1) < 5 or (new_y2 - new_y1) < 5:
            continue
        
        # Convert back to YOLO format
        new_center_x = (new_x1 + new_x2) / 2 / w
        new_center_y = (new_y1 + new_y2) / 2 / h
        new_width = (new_x2 - new_x1) / w
        new_height = (new_y2 - new_y1) / h
        
        new_bbox_line = f"{class_id} {new_center_x:.6f} {new_center_y:.6f} {new_width:.6f} {new_height:.6f}"
        new_bbox_lines.append(new_bbox_line)
    
    return rotated, new_bbox_lines

def flip_image_and_bboxes(image, bbox_lines, flip_code):
    flipped = cv2.flip(image, flip_code)
    
    new_bbox_lines = []
    
    for bbox_line in bbox_lines:
        parts = bbox_line.strip().split()
        if len(parts) != 5:
            continue
            
        class_id = parts[0]
        x_center, y_center, width, height = map(float, parts[1:])
        
        if flip_code == 1:  # Horizontal flip
            x_center = 1.0 - x_center
        elif flip_code == 0:  # Vertical flip
            y_center = 1.0 - y_center
        elif flip_code == -1:  # Both
            x_center = 1.0 - x_center
            y_center = 1.0 - y_center
        
        new_bbox_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        new_bbox_lines.append(new_bbox_line)
    
    return flipped, new_bbox_lines

def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def add_noise(image, noise_factor=0.1):
    noise = np.random.randint(0, int(255 * noise_factor), image.shape, dtype=np.uint8)
    noisy = cv2.add(image, noise)
    return noisy

def apply_augmentation(image, bbox_lines, aug_type):
    if aug_type == "rotate_15":
        return rotate_image_and_bboxes(image, bbox_lines, 15)
    elif aug_type == "rotate_-15":
        return rotate_image_and_bboxes(image, bbox_lines, -15)
    elif aug_type == "rotate_30":
        return rotate_image_and_bboxes(image, bbox_lines, 30)
    elif aug_type == "rotate_-30":
        return rotate_image_and_bboxes(image, bbox_lines, -30)
    elif aug_type == "flip_h":
        return flip_image_and_bboxes(image, bbox_lines, 1)
    elif aug_type == "flip_v":
        return flip_image_and_bboxes(image, bbox_lines, 0)
    elif aug_type == "bright":
        return adjust_brightness_contrast(image, alpha=1.2, beta=20), bbox_lines
    elif aug_type == "dark":
        return adjust_brightness_contrast(image, alpha=0.8, beta=-20), bbox_lines
    elif aug_type == "contrast":
        return adjust_brightness_contrast(image, alpha=1.3, beta=0), bbox_lines
    elif aug_type == "noise":
        return add_noise(image, 0.1), bbox_lines
    else:
        return image, bbox_lines


augmentation_types = [
    "rotate_15", "rotate_-15", "rotate_30", "rotate_-30",
    "flip_h", "flip_v", "bright", "dark", "contrast", "noise"
]

print("Starting data augmentation to balance classes...")

for class_name, current_count in classes_to_augment.items():
    needed = target_count - current_count
    print(f"\nAugmenting {class_name}: {current_count} -> {target_count} (+{needed} images)")
    
    class_folder = os.path.join(class_organized_dir, class_name)
    existing_images = [f for f in os.listdir(class_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    augmented_count = 0
    
    while augmented_count < needed:
        # Select random existing image
        source_image_name = random.choice(existing_images)
        source_image_path = os.path.join(class_folder, source_image_name)
        
        # Get corresponding label file
        label_name = os.path.splitext(source_image_name)[0] + ".txt"
        source_label_path = os.path.join(labels_dir, label_name)
        
        if not os.path.exists(source_label_path):
            continue
        
        # Read image and ALL label lines
        image = cv2.imread(source_image_path)
        if image is None:
            continue
            
        with open(source_label_path, 'r') as f:
            bbox_lines = f.readlines()
        
        # Remove empty lines
        bbox_lines = [line.strip() for line in bbox_lines if line.strip()]
        
        if not bbox_lines:
            continue
        
        # Apply random augmentation
        aug_type = random.choice(augmentation_types)
        augmented_image, augmented_bboxes = apply_augmentation(image, bbox_lines, aug_type)
        
        if not augmented_bboxes:  # Skip if no valid bboxes after augmentation
            continue
        
        # Generate new filename
        base_name = os.path.splitext(source_image_name)[0]
        new_image_name = f"{base_name}_aug_{augmented_count}_{aug_type}.jpg"
        new_label_name = f"{base_name}_aug_{augmented_count}_{aug_type}.txt"
        
        # Save augmented image
        new_image_path = os.path.join(images_dir, new_image_name)
        cv2.imwrite(new_image_path, augmented_image)
        
        # Save augmented labels (multiple lines)
        new_label_path = os.path.join(labels_dir, new_label_name)
        with open(new_label_path, 'w') as f:
            f.write('\n'.join(augmented_bboxes))
        
        # Copy to class-organized folder
        class_image_path = os.path.join(class_folder, new_image_name)
        shutil.copy2(new_image_path, class_image_path)
        
        augmented_count += 1
        
        if augmented_count % 50 == 0:
            print(f"  Generated {augmented_count}/{needed} augmented images for {class_name}")

print("complete")

# Count final distribution
final_counts = {}
for class_name in classes_to_augment.keys():
    class_folder = os.path.join(class_organized_dir, class_name)
    image_count = len([f for f in os.listdir(class_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
    final_counts[class_name] = image_count

print("\nFinal Class Distribution After Augmentation:")
for cls in classes_to_augment.keys():
    print(f"  {cls}: {final_counts[cls]} images")

total_images = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
total_labels = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

print(f"\nDataset Summary:")
print(f"  Total Images: {total_images}")
print(f"  Total Labels: {total_labels}")