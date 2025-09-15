import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import pandas as pd
import numpy as np
import math

def draw_bounding_boxes(image_obj, bounding_boxes, output_path=None, labels=None):
    """
    Draws bounding boxes on a PIL Image object.

    Args:
        image_obj (PIL.Image.Image): The Pillow Image object to draw on.
        bounding_boxes (list): A list of bounding box coordinates. Each box should be
                                a tuple (x_min, y_min, x_max, y_max).
        output_path (str, optional): The path to save the image with bounding boxes.
                                     If None, the image is not saved to disk.
        labels (list, optional): A list of strings, one label per bounding box.
                                 If provided, labels will be drawn near their boxes.

    Returns:
        PIL.Image.Image: The image with bounding boxes drawn, or None if an error occurs.
    """
    if not isinstance(image_obj, Image.Image):
        print("Error: Input is not a valid PIL Image object.")
        return None

    try:
        draw = ImageDraw.Draw(image_obj)

        # Try to load a default font. If it fails, use the default PIL font.
        try:
            # Common font paths on Linux/macOS. Adjust for Windows if needed.
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            print("Could not load 'arial.ttf', using default PIL font.")
            font = ImageFont.load_default()

        for i, box in enumerate(bounding_boxes):
            x_min, y_min, x_max, y_max = box
            # Draw the rectangle
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=3)

            # Draw label if provided
            if labels and i < len(labels):
                label_text = labels[i]
                # Determine text size
                text_bbox = draw.textbbox((0,0), label_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                # Position label above the box or inside if space is limited
                text_x = x_min
                text_y = y_min - text_height - 5 # 5 pixels padding above the box
                if text_y < 0: # If it goes off screen, put it inside
                    text_y = y_min + 5

                draw.text((text_x, text_y), label_text, fill="red", font=font)

        if output_path:
            image_obj.save(output_path)
            print(f"Image with bounding boxes saved to: {output_path}")

        return image_obj
    except Exception as e:
        print(f"An error occurred during drawing bounding boxes: {e}")
        return None



def apply_dino(self, image, text, model, processor, text_thresh=0.2, box_thresh=0.2):
    # applies grounding dino to image
    # example text: text = "a pillow. a chari."


    # bgr -> rgb
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    inputs = processor(images=pil_image, text=text, return_tensors="pt").to(self.device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_thresh,
        text_threshold=text_thresh,
        target_sizes=[pil_image.size[::-1]]
    )
    image_with_boxes = draw_bounding_boxes(pil_image, results[0]['boxes'], output_path="output_image_with_boxes.png", labels=results[0]['labels']) 
    fname = text.replace(" ", "_").replace(".", "")
    image_with_boxes.save("detected_"+fname+".jpg", quality=90) 
    return results


def make_depth_map(self):
    laser_cloud_xyz = self.lidar_cloud
    skip_conversion = False
    camera_offset_z = self.camera_offset_z
    PI = np.pi

    image_height, image_width, _ = self.image.shape
    image_depth = np.zeros((image_height, image_width), dtype=np.float32)
    image_depth.fill(np.nan)
    px_list = []
    py_list = []
    pz_list = []
    ix_list = []
    iy_list = []
    dist_list = []

    if (self.odom_id_pointer > 0) and (len(laser_cloud_xyz) != 0):
        # Synchronize odometry and image
        while (self.odom_time_stack[self.image_id_pointer] < self.image_time - 0.001 and self.image_id_pointer != (self.odom_id_pointer + 1) % self.stack_num):
            self.image_id_pointer = (self.image_id_pointer + 1) % self.stack_num

        if abs(self.odom_time_stack[self.image_id_pointer] - self.image_time) > 10:  #0.001:
            print("Odometry-Image time mismatch! Skipping frame.")
            skip_conversion = True

        if (skip_conversion == False):
            # Get synchronized LiDAR pose
            lidar_x = self.lidar_x_stack[self.image_id_pointer]
            lidar_y = self.lidar_y_stack[self.image_id_pointer]
            lidar_z = self.lidar_z_stack[self.image_id_pointer]
            lidar_roll = self.lidar_roll_stack[self.image_id_pointer]
            lidar_pitch = self.lidar_pitch_stack[self.image_id_pointer]
            lidar_yaw = self.lidar_yaw_stack[self.image_id_pointer]
 

            # Pre-calculate sines and cosines
            sin_lidar_roll = math.sin(lidar_roll)
            cos_lidar_roll = math.cos(lidar_roll)
            sin_lidar_pitch = math.sin(lidar_pitch)
            cos_lidar_pitch = math.cos(lidar_pitch)
            sin_lidar_yaw = math.sin(lidar_yaw)
            cos_lidar_yaw = math.cos(lidar_yaw)

            # Process each point in the LiDAR cloud
            for i in range(len(laser_cloud_xyz)):
                px, py, pz = laser_cloud_xyz[i]

                # Translate to LiDAR origin
                x1 = px - lidar_x
                y1 = py - lidar_y
                z1 = pz - lidar_z

                # Apply yaw rotation
                x2 = x1 * cos_lidar_yaw + y1 * sin_lidar_yaw
                y2 = -x1 * sin_lidar_yaw + y1 * cos_lidar_yaw
                z2 = z1

                # Apply pitch rotation
                x3 = x2 * cos_lidar_pitch - z2 * sin_lidar_pitch
                y3 = y2
                z3 = x2 * sin_lidar_pitch + z2 * cos_lidar_pitch

                # Apply roll rotation and camera Z offset
                x4 = x3
                y4 = y3 * cos_lidar_roll + z3 * sin_lidar_roll
                z4 = -y3 * sin_lidar_roll + z3 * cos_lidar_roll - camera_offset_z

                # Project to image plane
                hori_dis = math.sqrt(x4 * x4 + y4 * y4)
                if hori_dis == 0: # Avoid division by zero
                    continue

                hori_pixel_id = int(-image_width / (2 * PI) * math.atan2(y4, x4) + image_width / 2 + 1)
                vert_pixel_id = int(-image_width / (2 * PI) * math.atan(z4 / hori_dis) + image_height / 2 + 1) # Note: Was imageWidth, changed to imageHeight for vert_pixel_id

                # Check bounds
                if 0 <= hori_pixel_id < image_width and 0 <= vert_pixel_id < image_height:
                    distance = np.linalg.norm(np.array([x4, y4, z4]) - np.array([0, 0, 0]))
                    image_depth[vert_pixel_id, hori_pixel_id] = distance
                    px_list.append(px)
                    py_list.append(py)
                    pz_list.append(pz)
                    ix_list.append(hori_pixel_id)
                    iy_list.append(vert_pixel_id)
                    dist_list.append(distance)

            # end for loop

    pt2img_lut = pd.DataFrame({"px":px_list, "py":py_list, "pz":pz_list, "ix":ix_list, "iy":iy_list, "dist":dist_list})
    return image_depth, pt2img_lut


def pixel2direction(x_pixel, image_width):
    """
    Converts a pixel's horizontal location on a panoramic image to a direction (theta).

    The panoramic image is assumed to represent a full 360-degree view (2*pi radians).
    The leftmost pixel (x=0) corresponds to 0 radians, and the rightmost pixel
    (x=image_width-1) corresponds to just under 2*pi radians.

    Args:
        x_pixel (int): The horizontal pixel coordinate.
        image_width (int): The total width of the panoramic image in pixels.

    Returns:
        float: The direction in radians, where 0 <= theta < 2*pi.
    """

    if not (0 <= x_pixel < image_width):
        raise ValueError("x_pixel must be within the image width.")

    theta = (x_pixel / image_width) * 2 * np.pi

    return theta
