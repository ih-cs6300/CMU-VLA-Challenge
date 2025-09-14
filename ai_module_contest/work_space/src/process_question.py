import torch
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#from matplotlib import pyplot as plt, patches
from prompts import numerical_prompt, object_extraction_prompt, object_reference_prompt, instruction_following_prompt2, get_goal_position_prompt, instruction_following_prompt3
from prompts import cross_reference_prompt, cross_ref_coords_prompt
from gemini_stuff import ask_gemini
from path_utils import world2grid, grid2world, find_nearest_free, bresenham_line, reduce_path
import json
import cv2
import re
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
    ans, response_str = ask_gemini(filled_prompt, None, statement_type)
    return ans


def get_instr(instruction):
    regex_list = [
            r"goto\((.*), (.*)\)",
            r"^between\(\((.*), (.*)\), \((.*),(.*)\)\)",
            r"stop_at\((.*), (.*)\)",
            r"^avoid_between\(\((.*), (.*)\), \((.*),(.*)\)\)"
    ]
    instr_names = ["goto", "between", "stop_at", "avoid_between"]
    instr_args = None

    match = None
    idx = -1
    while (match is None) and (idx < len(regex_list)):
        idx += 1
        match = re.search(regex_list[idx], instruction)

    if (idx == 0) or (idx == 2):
        instr_args = (float(match.group(1)), float(match.group(2)))
    elif (idx == 1) or (idx == 3):
        instr_args = [(float(match.group(1)), float(match.group(2))), (float(match.group(3)), float(match.group(4)))]

    return instr_names[idx], instr_args

def cross_reference(gdf, challenge_question):
    # cross references objects mentioned in user statement with the names of objects detected in room
    room_objects = sorted(gdf.object_name.unique().tolist())
    room_objects_str = ", ".join(room_objects)
    filled_prompt = cross_reference_prompt.format(
        room_objects=room_objects_str,
        instr=challenge_question
    )
    ans, response_str = ask_gemini(filled_prompt, None, 'cross-reference')
    print("cross ref\n")
    print(filled_prompt)
    print(f"\n\nanswer: {ans}\n\n")
    return ans


def follow_instructions(
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
        statement_type='instruction-following'
    ):

    # convert instructions to program
    pprint_curr_loc = f"({curr_position[0]:.1f}, {curr_position[1]:.1f})"
    pprint_room_limits = f"(({MIN_X:.1f}, {MIN_Y:.1f}), ({MAX_X:.1f}, {MAX_Y:.1f}))"
    filled_prompt = instruction_following_prompt3.format(
            objs=json.dumps(obj_coord_center_list),
            curr_position=pprint_curr_loc,
            room_limits=pprint_room_limits,
            challenge_question=challenge_question
    )
    #ans, response_str = ask_openai(filled_prompt, 'target-coordinates')
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
        instr_name, instr_args = get_instr(curr_instr)

        if (instr_name == "goto") or (instr_name == 'stop_at'):
            # get start grid idx
            start = world2grid(xedges, yedges, pos)

            # get goal grid idx; make sure goal is traversable
            dist, pt_idx = kdtree.query(instr_args)
            goal = world2grid(xedges, yedges, point_matrix[pt_idx, :2])

            # check if goal is occupied square
            if counts[goal[0], goal[1]] == 1.:
                goal = find_nearest_free(goal, counts)

            # get path with A-star
            path = AStar(counts.astype(np.int8).tolist()).search(start, goal)
            path = [] if path is None else path
            complete_path += path
            pos = grid2world(xedges, yedges, path[-1])
        elif (instr_name == "between"):
            # get start idx
            start = world2grid(xedges, yedges, pos)

            # goal is mid-point of args
            args_arr = np.array(instr_args)
            p1 = np.mean(args_arr, axis=0)
            dist, pt_idx = kdtree.query(p1)
            goal = world2grid(xedges, yedges, point_matrix[pt_idx, :2])

            # check if goal is occupied square
            if counts[goal[0], goal[1]] == 1.:
                goal = find_nearest_free(goal, counts)

            path = AStar(counts.astype(np.int8).tolist()).search(start, goal)
            path = [] if path is None else path
            complete_path += path
            pos = grid2world(xedges, yedges, path[-1])
        elif (instr_name == "avoid_between"):
            # add obstacles to counts
            avoid1 = world2grid(xedges, yedges, instr_args[0])
            avoid2 = world2grid(xedges, yedges, instr_args[1])
            squares2set = bresenham_line(avoid1[0], avoid1[1], avoid2[0], avoid2[1])
            for sqr in squares2set:
                counts[sqr[0], sqr[1]] = 1
        else:
            raise ExceptionType("bad instruction name!")
        idx += 1

    complete_world_path = grid2world(xedges, yedges, complete_path) 

    processed_path = reduce_path(complete_world_path)

    return processed_path

def get_qtype_objects(self, challenge_question):
    # get objects of interest from challenge_question
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

    for stuff in obj_list:
        if ('round' in obj_dict['objects'][stuff]['physical-characteristics']) and (stuff == 'table'):
            obj_list.remove('table')
            obj_list.append('round table')
            break

    cross_referenced_obj_list = cross_reference(gdf, self.challenge_question)
    # create bounding box
    room_limits = gdf.total_bounds

    gdf = gdf.loc[gdf.object_name.isin(cross_referenced_obj_list), :]

    object_coord_center_list = create_object_center_list(gdf)
    gdf['center'] = gdf.geometry
    gdf['geometry'] = gdf.apply(lambda x: create_bbox(x['geometry'], x['scale']), axis=1)

    # create obstacle list for prompt
    object_coord_list = create_object_list(gdf)

    # answer numerical questions
    if (statement_type == 'numerical'):
       filled_prompt = numerical_prompt.format(obs=json.dumps(object_coord_list), challenge_question=challenge_question)
       print(filled_prompt)
       #ans, response_str = ask_openai(filled_prompt, statement_type)
       ans, response_str = ask_gemini(filled_prompt, None, statement_type)
       print(f"\n\nanswer: {ans['answer']}\n\n")
       self.answer = ans['answer']
    elif (statement_type == 'object-reference'):
       filled_prompt = object_reference_prompt.format(obs=json.dumps(object_coord_list), challenge_question=challenge_question)
       print(filled_prompt)
       #ans, response_str = ask_openai(filled_prompt, statement_type)
       ans, response_str = ask_gemini(filled_prompt, None, statement_type)
       obj_name = next(iter(ans))
       ans[obj_name] = [tuple(x) for x in ans[obj_name]]   # convert bbox corners from list to tuples
       row_num = object_coord_list.index(ans)  # get index of ans in gdf
       print('Answer:')
       print(gdf.iloc[row_num, gdf.columns.tolist().index('center')])
       self.answer = gdf.iloc[row_num, gdf.columns.tolist().index('center')]
    elif (statement_type == 'instruction-following'):
       route = follow_instructions(obj_dict, object_coord_list, object_coord_center_list, START_POSITION, point_matrix, gdf, MIN_X, MAX_X, MIN_Y, MAX_Y, challenge_question)
       self.answer = route

    self.got_answer = True

