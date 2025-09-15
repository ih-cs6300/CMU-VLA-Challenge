import torch
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
import json
import cv2
import re
import copy

from prompts import numerical_prompt, object_extraction_prompt, object_reference_prompt, instruction_following_prompt2, get_goal_position_prompt, instruction_following_prompt4
from prompts import cross_reference_prompt, cross_ref_coords_prompt
from gemini_stuff import ask_gemini
from path_utils import world2grid, grid2world, find_nearest_free, bresenham_line, reduce_path, clean_path, smooth_path
from object_detection import apply_dino, pixel2direction
from publish_answers import publish_waypoints

from astar.search import AStar
from scipy.spatial import KDTree

def create_bbox(centroid, scale):
   x_list = [(centroid.x-scale[0]/2), (centroid.x+scale[0]/2), (centroid.x+scale[0]/2), (centroid.x-scale[0]/2)]
   y_list = [(centroid.y+scale[1]/2), (centroid.y+scale[1]/2), (centroid.y-scale[1]/2), (centroid.y-scale[1]/2)]
   z_list = [(centroid.z-scale[2]/2), (centroid.z+scale[2]/2), (centroid.z+scale[2]/2), (centroid.z-scale[2]/2)]

   bbox = Polygon(zip(x_list, y_list))
   return bbox

def create_object_list(object_df):
    obstacle_list = []
    for idx, row in object_df.iterrows():
        minx, miny, maxx, maxy = row['geometry'].bounds
        temp_dict = {row['object_name']: [(round(minx, 1), round(miny, 1)), (round(maxx, 1), round(maxy, 1))]}
        obstacle_list.append(temp_dict)

    return obstacle_list

def create_object_center_list(object_df):
    obj_list = []
    for idx, row in object_df.iterrows():
        temp_dict = {row['object_name']: (round(row['geometry'].x, 1), round(row['geometry'].y, 1))}
        obj_list.append(temp_dict)
    return obj_list


def extract_objects(msg, statement_type):
    """
    Extracts the objects from question.
    """
    filled_prompt = object_extraction_prompt.format(challenge_question=msg)
    #ans, resp = ask_openai(filled_prompt, statement_type)
    print(f"  ask: llm obj/extraction-qtype")
    ans, response_str = ask_gemini(filled_prompt, None, statement_type) 
    return ans


def get_instr(instruction,  use_obj_coords=True):
    regex_list1 = [
            r"goto\((.*), (.*)\)",
            r"^between\(\((.*), (.*)\), \((.*),(.*)\)\)",
            r"stop_at\((.*), (.*)\)",
            r"^avoid_between\(\((.*), (.*)\), \((.*),(.*)\)\)"
    ]
    regex_list2 = [
            r"goto\((.*)\)",
            r"^between\((.*), (.*)\)",
            r"stop_at\((.*)\)",
            r"^avoid_between\((.*), (.*)\)"

    ]
    
    regex_list = regex_list1 if use_obj_coords else regex_list2

    instr_names = ["goto", "between", "stop_at", "avoid_between"]
    instr_args = None

    match = None
    idx = -1
    while (match is None) and (idx < len(regex_list)):
        idx += 1
        match = re.search(regex_list[idx], instruction)

    if (use_obj_coords == True):
        if (idx == 0) or (idx == 2):
            instr_args = (float(match.group(1)), float(match.group(2)))
        elif (idx == 1) or (idx == 3):
            instr_args = [(float(match.group(1)), float(match.group(2))), (float(match.group(3)), float(match.group(4)))]
    else:
       if (idx == 0) or (idx == 2):
           instr_args = match.group(1)
       elif (idx == 1) or (idx == 3):
           instr_args = [match.group(1), match.group(2)]

    return instr_names[idx], instr_args

def cross_reference(gdf, challenge_question):
    # cross references objects mentioned in user statement with the names of objects detected in room
    room_objects = sorted(gdf.object_name.unique().tolist())
    room_objects_str = ", ".join(room_objects)
    filled_prompt = cross_reference_prompt.format(
        room_objects=room_objects_str,
        instr=challenge_question
    )
    print("   ask: llm object cross reference")
    ans, response_str = ask_gemini(filled_prompt, None, 'cross-reference')
    print("cross ref\n")
    print(filled_prompt)
    print(f"\n\nanswer: {ans}\n\n")
    return ans

def process_obj_detection_res(image_depth, yaw, obj_det_res, image_ht, image_wd):
    bboxes = obj_det_res[0]['boxes']
    labels = obj_det_res[0]['labels']
    scores = obj_det_res[0]['scores']

    idx = torch.argmax(scores).item()
                                                                                                                  
    # get first bbox center
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = torch.round(bboxes[idx]).to(torch.int64)
    bbox_center = ((bbox_xmin+bbox_xmax)/2, (bbox_ymin+bbox_ymax)/2) 
    bbox_center = (bbox_center[0].item(), bbox_center[1].item())
                                                                                                                  
    # get objects distance from robot                         
    obj_depth = np.nanmean(image_depth[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax])
                                                                                                                  
    # get objects direction       
    theta_pix = pixel2direction(bbox_center[0], image_wd)   # 0 rads is at image left and 2pi rads is at image right
    theta_world = yaw + (np.pi - theta_pix)
    x_world = obj_depth * np.cos(theta_world)
    y_world = obj_depth * np.sin(theta_world)
    return (x_world, y_world)



def follow_instructions(
        self,
        obj_dict, 
        obj_coord_list, 
        obj_coord_center_list, 
        curr_position, 
        point_matrix, 
        gdf, 
        MIN_X, 
        MAX_X, 
        MIN_Y, 
        MAX_Y, 
        challenge_question, 
        image, 
        image_depth,
        pt2img_dist_lut, 
        obj_detection_model, 
        obj_detection_processor,
        yaw,
        way_pub,
        statement_type='instruction-following'
    ):

    # convert instructions to program
    #pprint_curr_loc = f"({curr_position[0]:.1f}, {curr_position[1]:.1f})"
    #pprint_room_limits = f"(({MIN_X:.1f}, {MIN_Y:.1f}), ({MAX_X:.1f}, {MAX_Y:.1f}))"
    filled_prompt = instruction_following_prompt4.format(challenge_question=challenge_question)
    n_pts = -1 # num of smoothed pts

    #ans, response_str = ask_openai(filled_prompt, 'target-coordinates')
    print("  ask: llm instr-to-prog")
    ans, response_str = ask_gemini(filled_prompt, None, statement_type)

    print(filled_prompt)
    print(response_str)
    print(ans)

    # get occupancy grid
    counts, xedges, yedges = np.histogram2d(point_matrix[:, 0], point_matrix[:, 1], bins=(np.ceil((MAX_X-MIN_X)/0.25).astype(np.int64), np.ceil((MAX_Y-MIN_Y)/0.25).astype(np.int64)))
    counts = np.where(counts > 10., 0., 1.)

    idx = 0
    #fig, ax = plt.subplots()
    complete_path = []
    pos = curr_position
    kdtree = KDTree(point_matrix[:, :2])

    # order instructions so that areas to avoid are first while preserving order of goto and between instructions
    avoid_instr = []
    for inst in ans:
        if "avoid" in inst:
            avoid_instr.append(inst)

    for inst in avoid_instr:
        ans.remove(inst)

    ans = avoid_instr + ans
    while idx < len(ans):
        curr_instr = ans[idx]
        instr_name, instr_args = get_instr(curr_instr, use_obj_coords=False)

        if (instr_name == "goto") or (instr_name == 'stop_at'):
            # get start grid idx
            start = world2grid(xedges, yedges, pos)

            # locate objects in image
            image_text = instr_args.replace("_", " ") + "."
            obj_det_res = apply_dino(self, image, image_text, obj_detection_model, obj_detection_processor, text_thresh=0.2, box_thresh=0.2)
            image_ht, image_wd, _ = image.shape
            arg_coords = process_obj_detection_res(image_depth, yaw, obj_det_res, image_ht, image_wd)    

            # get goal grid idx; make sure goal is traversable
            dist, pt_idx = kdtree.query(arg_coords)
            goal = world2grid(xedges, yedges, point_matrix[pt_idx, :2])

            # check if goal is occupied square
            if counts[goal[0], goal[1]] == 1.:
                goal = find_nearest_free(goal, counts)

            # get path with A-star
            path = AStar(counts.astype(np.int8).tolist()).search(start, goal)
            path = [] if path is None else path
            path_world_g = grid2world(xedges, yedges, path)
            red_path_g = clean_path(path_world_g, kdtree, point_matrix, xedges, yedges, counts)
            #red_path_g = smooth_path(red_path_g, n_pts, s_var=12)  # causes errors sometimes
            red_path_g = reduce_path(red_path_g)
            complete_path += red_path_g

            # move to new location
            publish_waypoints(self, red_path_g, way_pub)

            pos = red_path_g[-1]
        elif (instr_name == "between"):
            # get start idx
            start = world2grid(xedges, yedges, pos)

            # locate objects in image
            image_text1 = instr_args[0].replace("_", " ") + "."
            image_text2 = instr_args[1].replace("_", " ") + "."
            obj_det_res1 = apply_dino(self, image, image_text1, obj_detection_model, obj_detection_processor, text_thresh=0.2, box_thresh=0.2)
            obj_det_res2 = apply_dino(self, image, image_text2, obj_detection_model, obj_detection_processor, text_thresh=0.2, box_thresh=0.2)
            image_ht, image_wd, _ = image.shape
            arg_coords1 = process_obj_detection_res(image_depth, yaw, obj_det_res1, image_ht, image_wd)
            arg_coords2 = process_obj_detection_res(image_depth, yaw, obj_det_res2, image_ht, image_wd)


            # goal is mid-point of args
            args_arr = np.array([arg_coords1, arg_coords2])
            p1 = np.mean(args_arr, axis=0)

            dist, pt_idx = kdtree.query(p1)
            goal = world2grid(xedges, yedges, point_matrix[pt_idx, :2])

            # check if goal is occupied square
            if counts[goal[0], goal[1]] == 1.:
                goal = find_nearest_free(goal, counts)

            path = AStar(counts.astype(np.int8).tolist()).search(start, goal)
            path = [] if path is None else path
            path_world_b = grid2world(xedges, yedges, path)
            red_path_b = clean_path(path_world_b, kdtree, point_matrix, xedges, yedges, counts)
            #red_path_b = smooth_path(red_path_b, n_pts, s_var=4)   # causes errors sometimes
            red_path_b = reduce_path(red_path_b)
            complete_path += red_path_b

            # move to new location
            publish_waypoints(self, red_path_b, way_pub)

            pos = red_path_b[-1]
        elif (instr_name == "avoid_between"):
            # areas to avoid

            # locate objects in image                                                                                                          
            image_text1 = instr_args[0].replace("_", " ") + "."
            image_text2 = instr_args[1].replace("_", " ") + "."
            obj_det_res1 = apply_dino(self, image, image_text1, obj_detection_model, obj_detection_processor, text_thresh=0.2, box_thresh=0.2)
            obj_det_res2 = apply_dino(self, image, image_text2, obj_detection_model, obj_detection_processor, text_thresh=0.2, box_thresh=0.2)
            image_ht, image_wd, _ = image.shape
            arg_coords1 = process_obj_detection_res(image_depth, yaw, obj_det_res1, image_ht, image_wd)
            arg_coords2 = process_obj_detection_res(image_depth, yaw, obj_det_res2, image_ht, image_wd)
                                                                                                                                               
            avoid1 = world2grid(xedges, yedges, arg_coords1)
            avoid2 = world2grid(xedges, yedges, arg_coords2)
            squares2set = bresenham_line(avoid1[0], avoid1[1], avoid2[0], avoid2[1])
            for sqr in squares2set:
                counts[sqr[0], sqr[1]] = 1
        else:
            raise ExceptionType("bad instruction name!")
        idx += 1

    return complete_path

def get_qtype_objects(self, challenge_question):
    # get objects of interest from challenge_question
    if (self.statement_type == ""):
        obj_dict = extract_objects(challenge_question, 'object-extraction')
        statement_type = obj_dict['statment-type']

        self.statement_type = statement_type
        self.question_objects = [*obj_dict['objects']]
        self.obj_dict = obj_dict

    

def question_processor_driver(self):
    # variables for instruction following
    START_POSITION = (self.position_x, self.position_y)
    challenge_question = self.challenge_question
    point_matrix = self.point_matrix
    gdf = self.gdf

    MIN_X = np.min(point_matrix[:, 0])
    MAX_X = np.max(point_matrix[:, 0])
    MIN_Y = np.min(point_matrix[:, 1])
    MAX_Y = np.max(point_matrix[:, 1])

    # get objects of interest from challenge_question
    obj_dict = self.obj_dict
    statement_type = self.statement_type
    obj_list = self.question_objects

    obj_list_copy = copy.deepcopy(obj_list)
    try:
        for stuff in obj_list_copy:
            if ('round' in obj_dict['objects'][stuff]['physical-characteristics']) and (stuff == 'table'):
                obj_list.remove('table')
                obj_list.append('round table')
                break
    except Exception as e:
        print(f"An error occurred: {e}")  # TODO sometimes an error here; fix


    cross_referenced_obj_list = [] if self.statement_type == 'instruction-following' else cross_reference(gdf, self.challenge_question)

    # create bounding box
    #room_limits = gdf.total_bounds
    
    gdf = gdf.loc[gdf.object_name.isin(cross_referenced_obj_list), :]
    object_coord_center_list = create_object_center_list(gdf)

    try:
        gdf['center'] = gdf.geometry
        gdf['geometry'] = gdf.apply(lambda x: create_bbox(x['geometry'], x['scale']), axis=1)
    except Exception as e:
        print(f"Error: {e}")

    # create obstacle list for prompt
    object_coord_list = create_object_list(gdf)

    # answer numerical questions
    if (statement_type == 'numerical'):
       filled_prompt = numerical_prompt.format(obs=json.dumps(object_coord_list), challenge_question=challenge_question)
       print(filled_prompt)
       #ans, response_str = ask_openai(filled_prompt, statement_type)
       print("  ask: llm numerical question")
       ans, response_str = ask_gemini(filled_prompt, None, statement_type)
       print(f"\n\nanswer: {ans['answer']}\n\n")
       self.answer = ans['answer']
    elif (statement_type == 'object-reference'):
       filled_prompt = object_reference_prompt.format(obs=json.dumps(object_coord_list), challenge_question=challenge_question)
       print(filled_prompt)
       #ans, response_str = ask_openai(filled_prompt, statement_type)
       print("  ask:  llm obj-ref-question")
       ans, response_str = ask_gemini(filled_prompt, None, statement_type)
       obj_name = next(iter(ans))
       ans[obj_name] = [tuple(x) for x in ans[obj_name]]   # convert bbox corners from list to tuples
       row_num = object_coord_list.index(ans)  # get index of ans in gdf
       print('Answer:')
       print(gdf.iloc[row_num, gdf.columns.tolist().index('center')])
       self.answer = gdf.iloc[row_num, gdf.columns.tolist().index('center')]
    elif (statement_type == 'instruction-following'):
       route = follow_instructions(self, obj_dict, object_coord_list, object_coord_center_list, START_POSITION, point_matrix, gdf, MIN_X, MAX_X, MIN_Y, MAX_Y, challenge_question, 
                                   self.image, self.image_depth, self.pt2img_dist_lut, self.obj_detection_model, self.obj_detection_processor, self.yaw, self.waypoint_pub)
       self.answer = route
       self.done_exploring = True

    self.got_answer = True

