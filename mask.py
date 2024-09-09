import json
from os import path as osp
import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from PIL import ImageChops


def get_img_agnostic(img, parse, pose_data, hand_data):
    parse_array = np.array(parse)
    # Grouping the parsed area
    parse_head = ((parse_array == 4).astype(np.float32) + # sunglasses
                    (parse_array == 13).astype(np.float32) + # face
                    (parse_array == 2).astype(np.float32) + # hair
                    (parse_array == 1).astype(np.float32) # hat
                    )
    
    parse_lower = ((parse_array == 9).astype(np.float32) + # pants
                    (parse_array == 12).astype(np.float32) + # skirt
                    (parse_array == 16).astype(np.float32) + # left leg
                    (parse_array == 17).astype(np.float32) + # right leg
                    (parse_array == 18).astype(np.float32) + # left shoe
                    (parse_array == 19).astype(np.float32)) # right shoe
    
    parse_upper = ((parse_array == 5).astype(np.float32) + # upper clothes
                   (parse_array == 7).astype(np.float32) + # coat
                   (parse_array == 11).astype(np.float32) +  # Scarf
                   (parse_array == 10).astype(np.float32) + # tosor skin
                   (parse_array == 14).astype(np.float32) + # Left arm
                   (parse_array == 15).astype(np.float32))  # right arm
    
    parse_hand = ((parse_array == 14).astype(np.float32) + # left arm
                    (parse_array == 15).astype(np.float32) + # right arm
                    (parse_array == 3).astype(np.float32)  # glove
                    )
    # draw the original image
    agnostic = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    grey = Image.new('L', img.size, color=128)
    upper_mask = Image.fromarray(np.uint8(parse_upper * 255), 'L') # select upper part of the body
    grey_upper = Image.composite(grey, img, upper_mask)
    agnostic.paste(grey_upper, (0, 0), upper_mask)
    upper_mask_array = np.array(upper_mask)


    # Find coordinates of the upper part of the body
    y_coords, x_coords = np.nonzero(upper_mask_array)
    
    if x_coords.size > 0 and y_coords.size > 0:
        # Find maximum and minimum coordinates
        min_x = np.min(x_coords)
        max_x = np.max(x_coords)
        min_y = np.min(y_coords)
        max_y = np.max(y_coords)
        
        # Draw grey rectangle around the min and max points
        agnostic_draw.rectangle([min_x, min_y, max_x, max_y], fill='grey', outline=None)
    else:
        print("Upper body mask has no non-zero pixels.")

    # Hand Exclusion
    if isinstance(hand_data, np.ndarray) and hand_data.ndim == 3:
        
        for hand in hand_data:
            points = [point for point in hand if not (point[0] == 0 and point[1] == 0)]

            if len(points) > 1:
                points = np.array(points) # Hand data into array
                first_point = points[0] # find the wrist of the hand
                distances = [np.linalg.norm(np.array(point) - first_point) for point in points] # Distance between the wrist and all the other hand points
                max_dis = np.max(distances) # Find max distance

                # Find the point with the max distance
                furthest_point_index = np.argmax(distances)
                furthest_point = np.array(points[furthest_point_index])
                    
                # find the center point of the hand, half way through the max distance
                hand_x = (first_point[0] + furthest_point[0]) / 2
                hand_y = (first_point[1] + furthest_point[1]) / 2

                # Radius is half the max distance. Set to 0.6 to make sure the majority of the hand is included
                hand_radius = 0.6 * max_dis

                # Drawing a circle around hand position
                hand_mask = Image.new('L', img.size, 0)
                hand_draw = ImageDraw.Draw(hand_mask)
                hand_draw.ellipse([(hand_x - hand_radius, hand_y - hand_radius), 
                                    (hand_x + hand_radius, hand_y + hand_radius)], 
                                    fill = 255                                
                                    )
                
                # Parsed data of the arm
                parse_hand = np.array(parse_hand) 
                parse_hand_mask = Image.fromarray(np.uint8(parse_hand * 255), 'L').convert('1') 
                hand_mask = hand_mask.convert('1')
                parse_hand_mask = parse_hand_mask.convert('1')

                # Find intersection of the parsed data and circle
                intersection_mask = ImageChops.logical_and(hand_mask, parse_hand_mask)

                agnostic.paste(img, (0, 0), intersection_mask)

    # Paste the head and lower body part on top of the grey area.
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

    return agnostic

def get_img_agnostic_bottom(img, parse, pose_data):
    parse_array = np.array(parse)
  
    #parsed results 
    parse_head = ((parse_array == 4).astype(np.float32) + # sunglasses
                    (parse_array == 13).astype(np.float32) +  # face
                    (parse_array == 2).astype(np.float32) + # hair
                    (parse_array == 1).astype(np.float32) # hat
                    )
    
    parse_lower = ((parse_array == 9).astype(np.float32) + # pants      
                    (parse_array == 12).astype(np.float32) + # skirt
                    (parse_array == 16).astype(np.float32) + # left leg
                    (parse_array == 17).astype(np.float32) # right leg
                    )
       
    parse_upper = ((parse_array == 5).astype(np.float32) + # upper cloeths
                    (parse_array == 8).astype(np.float32) + # socks
                    (parse_array == 7).astype(np.float32) + # coat
                    (parse_array == 11).astype(np.float32) +  # Scarf
                    (parse_array == 10).astype(np.float32) + # tosor-skin
                    (parse_array == 14).astype(np.float32) + # Left arm
                    (parse_array == 15).astype(np.float32) + # right arm
                    (parse_array == 18).astype(np.float32) + # left shoe
                    (parse_array == 19).astype(np.float32) # right show
                   ) 
    
    parse_hand = ((parse_array == 14).astype(np.float32) + # left arm
                    (parse_array == 15).astype(np.float32) + # right arm
                    (parse_array == 3).astype(np.float32) # glove
                    )
    
    agnostic_bottom = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic_bottom)


    grey = Image.new('L', img.size, color=128)
    lower_mask = Image.fromarray(np.uint8(parse_lower * 255), 'L') # select lower part of body
    grey_lower = Image.composite(grey, img, lower_mask) # colour it grey
    agnostic_bottom.paste(grey_lower, (0, 0), lower_mask)
    
    lower_mask_array = np.array(lower_mask)
    y_coords, x_coords = np.nonzero(lower_mask_array) # get coordinates of lower part of the body
    
    if x_coords.size > 0 and y_coords.size > 0:
        # Find maximum and minimum points
        min_x = np.min(x_coords)
        max_x = np.max(x_coords)
        min_y = np.min(y_coords)
        max_y = np.max(y_coords)
        
        agnostic_draw.rectangle([min_x, min_y, max_x, max_y], fill='grey', outline=None) # draw rectangle
    else:
        print("Lower body mask has no non-zero pixels.")


    # Hand Exclusion
    if isinstance(hand_data, np.ndarray) and hand_data.ndim == 3:
        
        for hand in hand_data:
            points = [point for point in hand if not (point[0] == 0 and point[1] == 0)]

            if len(points) > 1:
                points = np.array(points) # Hand data into array
                first_point = points[0] # find the wrist of the hand
                distances = [np.linalg.norm(np.array(point) - first_point) for point in points] # Distance between the wrist and all the other hand points
                max_dis = np.max(distances) # Find max distance

                # Find the point with the max distance
                furthest_point_index = np.argmax(distances)
                furthest_point = np.array(points[furthest_point_index])
                    
                # find the center point of the hand, half way through the max distance
                hand_x = (first_point[0] + furthest_point[0]) / 2
                hand_y = (first_point[1] + furthest_point[1]) / 2

                # Radius is half the max distance. Set to 0.6 to make sure the majority of the hand is included
                hand_radius = 0.6 * max_dis

                # Drawing a circle around hand position
                hand_mask = Image.new('L', img.size, 0)
                hand_draw = ImageDraw.Draw(hand_mask)
                hand_draw.ellipse([(hand_x - hand_radius, hand_y - hand_radius), 
                                    (hand_x + hand_radius, hand_y + hand_radius)], 
                                    fill = 255                                
                                    )
                
                # Parsed data of the arm
                parse_hand = np.array(parse_hand) 
                parse_hand_mask = Image.fromarray(np.uint8(parse_hand * 255), 'L').convert('1') 
                hand_mask = hand_mask.convert('1')
                parse_hand_mask = parse_hand_mask.convert('1')

                # Find intersection of the parsed data and circle
                intersection_mask = ImageChops.logical_and(hand_mask, parse_hand_mask)

                agnostic_bottom.paste(img, (0, 0), intersection_mask)

    agnostic_bottom.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic_bottom.paste(img, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    
    return agnostic_bottom

if __name__ == "__main__":
    data_path = './Input'
    pose_path = './Output/pose'
    parse_path = './Output/parse'
    output_path = './Output/agnostic'
    output_bottom_path = './Output/agnostic_bottom'


    os.makedirs(output_path, exist_ok=True)
    
    for im_name in tqdm(os.listdir(data_path)):

        pose_name = im_name.replace('.jpg', '_keypoints.json')
        handpoints = im_name.replace('.jpg', '_handpoints.json')

        # Load pose points
        try:
            with open(osp.join(pose_path, 'json', pose_name), 'r') as f:
                pose_label = json.load(f)
                
                if 'candidate' in pose_label:
                    pose_data = pose_label['candidate']
                    pose_data = np.array([[point[0], point[1]] for point in pose_data])

                else:
                    print(f"'candidate' key not found in {pose_name}")
                    continue
                    
        except (IndexError, KeyError, ValueError, TypeError) as e:
            print(f"Error processing {pose_name}: {e}")
            continue
        
        #Load handpoints
        try:
            with open(osp.join(pose_path, 'json', handpoints), 'r') as f:
                hand_label = json.load(f)
                
                # Extract keypoints from 'candidate'
                if 'hand_peaks' in hand_label:
                    hand_data = hand_label['hand_peaks']
                    hand_data = np.array(hand_data)
                                  
                else:
                    print(f"'hand_peaks' key not found in {handpoints}")
                    continue
                    
        except (IndexError, KeyError, ValueError, TypeError) as e:
            print(f"Error processing {handpoints}: {e}")
            continue

        # Load parsing image
        im = Image.open(osp.join(data_path, im_name))
        label_name = im_name.replace('.jpg', '_label.png')
        im_label = Image.open(osp.join(parse_path, label_name))

        agnostic = get_img_agnostic(im, im_label, pose_data, hand_data)
        
        agnostic.save(osp.join(output_path, im_name))

        agnostic_bottom = get_img_agnostic_bottom(im, im_label, pose_data)
        
        agnostic_bottom.save(osp.join(output_bottom_path, im_name))