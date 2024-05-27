import os
import cv2
import numpy as np
import yaml

def load_yaml_labels(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
        return data.get('names', {})

def extract_points_from_string(line, shape):
    values = line.strip().split(' ')
    label = values[0]
    points = [(float(values[i]) * shape[1], float(values[i + 1]) * shape[0]) for i in range(1, len(values) - 1, 2)]
    return points, label

def extract_points_from_file(file_path, shape):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    all_points = []
    labels = []
    for line in lines:
        points, label = extract_points_from_string(line, shape)
        all_points.append(points)
        labels.append(label)
    return all_points, labels

def clip_and_resize_image(image, points):
    mask = np.zeros_like(image, dtype=np.uint8)
    points = np.array([points], dtype=np.int32)
    cv2.fillPoly(mask, points, (255, 255, 255))
    clipped_image = cv2.bitwise_and(image, mask)
    
    x_coords = [p[0] for p in points[0]]
    y_coords = [p[1] for p in points[0]]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_max)
    y_max = min(image.shape[0], y_max)

    if x_max > x_min and y_max > y_min:
        cropped_image = clipped_image[y_min:y_max, x_min:x_max]
        return cropped_image
    else:
        print(f"Invalid bounding box: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
        return None


input_folder = "Text Detection.v2i.yolov8-obb/test"
output_folder = "Text Detection.v2i.yolov8-obb/clip1"
yaml_file_path = "Text Detection.v2i.yolov8-obb/data.yaml"


labels_dict = load_yaml_labels(yaml_file_path)


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_folder, filename)
        image_filename = filename.replace(".txt", ".jpg")
        image_path = os.path.join(input_folder, image_filename)
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            continue
        
        all_points, label_ids = extract_points_from_file(file_path, image.shape)
        
        for i, (points, label_id) in enumerate(zip(all_points, label_ids)):
            output_filename = f"{os.path.splitext(image_filename)[0]}_{i}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            clipped_image = clip_and_resize_image(image, points)
            
            if clipped_image is not None and clipped_image.size > 0:
                cv2.imwrite(output_path, clipped_image)
                print(f"Image {i} clipped and saved as {output_path}")
                label_name = labels_dict.get(int(label_id), "unknown")
                
                label_filename = f"{os.path.splitext(image_filename)[0]}_{i}.txt"
                label_path = os.path.join(output_folder, label_filename)
                with open(label_path, 'w') as label_file:
                    label_file.write(f"{output_filename}, {label_name}")
                print(f"Label saved as {label_path}")
            else:
                print(f"Skipping saving for {output_filename} due to invalid cropped image")

print("All images clipped and saved successfully.")
