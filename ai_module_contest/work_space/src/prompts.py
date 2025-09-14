object_extraction_prompt = (
"""
You are a robot interacting with a human user in an indoor environment.  All user statements will fall exclusively into 1 of 3 categories: numerical, instruction-following, or object-reference.  Numerical statements are questions that ask the number of objects in the environment with certain physical or spatial attributes.  Object-reference statements instruct you to locate an object in the environment.  Instruction-following statements instruct you navigate to a target location using objects in the room as references.  All statements will reference objects in the environment.       

Given a user statement, classify it as one of three types: "numerical",  "object-reference", or "instruction-following".  List each object referenced in the statement in its singular form, along with modifying information of interest.  Each statement will have a target object.  The target object is the object to be counted, found, or the final destination in a path.  Give the target object as well as the other information as a JSON string that follows the format shown in the examples.  

Examples:
   Statement:
      How many blue chairs are between the table and the wall?
   Answer: 
      {{
        "statment-type": "numerical",
        "target-object": "chair",
        "objects": {{
          "chair": {{
            "physical-characteristics": [
              "blue"
            ],
            "spatial-characteristics": {{
              "between": [
                "table",
                "wall"
              ]
            }}
          }},
          "table": {{
            "physical-characteristics": [],
            "spatial-characteristics": {{}}
          }},
          "wall": {{
            "physical-characteristics": [],
            "spatial-characteristics": {{}}
          }}
        }}
      }}
   Statement:
      How many black trash cans are near the window?
   Answer:
      {{
        "statment-type": "numerical",
        "target-object": "trash can",
        "objects": {{
          "trash can": {{
            "physical-characteristics": [
              "black"
            ],
            "spatial-characteristics": {{
              "near": [
                "window"
              ]
            }}
          }},
          "window": {{
            "physical-characteristics": [],
            "spatial-characteristics": {{}}
          }}
        }}
      }}
   Statement:
      Take the path near the window to the fridge.
   Answer:
      {{
        "statment-type": "instruction-following",
        "target-object": "fridge",
        "objects": {{
          "window": {{
            "physical-characteristics": [],
            "spatial-characteristics": {{}}
          }},
          "fridge": {{
            "physical-characteristics": [],
            "spatial-characteristics": {{}}
          }}
        }}
      }}
   Statement:
      Avoid the path between the two tables and go near the blue trash can near the window.
   Answer:
      {{
        "statment-type": "instruction-following",
        "target-object": "trash can",
        "objects": {{
          "table": {{
            "physical-characteristics": [],
            "spatial-characteristics": {{}}
          }},
          "trash can": {{
            "physical-characteristics": [
              "blue"
            ],
            "spatial-characteristics": {{
              "near": [
                "window"
              ]
            }}
          }},
          "window": {{
            "physical-characteristics": [],
            "spatial-characteristics": {{}}
          }}
        }}
      }}
   Statement:
      Find the potted plant on the kitchen island that is closest to the fridge.
   Answer:
      {{
        "statment-type": "object-reference",
        "target-object": "plant",
        "objects": {{
          "plant": {{
            "physical-characteristics": [
              "potted"
            ],
            "spatial-characteristics": {{
              "on": [
                "kitchen island"
              ],
              "closest": [
                "fridge"
              ]
            }}
          }},
          "kitchen island": {{
            "physical-characteristics": [],
            "spatial-characteristics": {{}}
          }},
          "fridge": {{
            "physical-characteristics": [],
            "spatial-characteristics": {{}}
          }}
        }}
      }}
   Statement:
      Find the orange chair between the table and sink that is closest to the window.
   Answer:
      {{
        "statment-type": "object-reference",
        "target-object": "chair",
        "objects": {{
          "chair": {{
            "physical-characteristics": [
              "orange"
            ],
            "spatial-characteristics": {{
              "between": [
                "table",
                "sink"
              ],
              "closest": [
                "window"
              ]
            }}
          }},
          "table": {{
            "physical-characteristics": [],
            "spatial-characteristics": {{}}
          }},
          "sink": {{
            "physical-characteristics": [],
            "spatial-characteristics": {{}}
          }},
          "window": {{
            "physical-characteristics": [],
            "spatial-characteristics": {{}}
          }}
        }}
      }}
Statement:
   {challenge_question}
Answer:
"""
)

numerical_prompt = (
"""
You are a wheeled robot in an indoor environment.  The room has been mapped to a x-y world grid.  Using your array of sensors you have detected the locations of important objects in the room.  The location of each object is demarcated by a bounding box.  The bounding boxes are defined using JSON dictionaries as follows, {{"object name": [[min_x, min_y], [max_x, max_y]]}} where the key is the object's name and the value is a list of two tuples.  The first tuple is the bounding box minimum x and minimum y and the second tuple is the bounding box maximum x and maximum y.

Here is the list of objects: {obs}.  Use the object list and spatial reasoning to answer the question.

Response Format:
    Provide the answer as a single integer.
    Please ONLY respond in the format:
        Reasoning: reason about the given answer and
        Answer: your answer


Question:
   {challenge_question}
"""
)

object_reference_prompt = (
"""
You are a wheeled robot in an indoor environment.  The room has been mapped to a x-y world grid.  Using your array of sensors you have detected the locations of important objects in the room.  The location of each object is demarcated by a bounding box.  The bounding boxes are defined using JSON dictionaries as follows, {{"object name": [[min_x, min_y], [max_x, max_y]]}} where the key is the object's name and the value is a list of two tuples.  The first tuple is the bounding box minimum x and minimum y and the second tuple is the bounding box maximum x and maximum y.

Here is the list of objects: {obs}.  Use the object list and spatial reasoning to answer the question.  Answer with the object's name and bounding box as provided in the objects list.

Response Format:
    Provide the answer as a JSON dictionary 
    Please ONLY respond in the format:
        Reasoning: reason about the given answer and
        Answer: {{"object name": [[x1, y1], [x2, y2]]}}


Question:
   {challenge_question}
"""
)

instruction_following_prompt1 = (
"""
You are a wheeled robot in a 3D indoor environment.  The room has been mapped to a x-y world grid.  Using your array of sensors you have detected the locations of important objects in the room.  The location of each object is demarcated by a bounding box.  The bounding boxes are defined using JSON dictionaries as follows, {{"object name": [[min_x, min_y], [max_x, max_y]]}} where the key is the object's name and the value is a list of two tuples.  The first tuple is the bounding box minimum x and minimum y and the second tuple is the bounding box maximum x and maximum y.  

Here is the list of objects: {obs}. 

Here is your movement history: [{pprint_path}].

You are currently at {pprint_curr_loc}.

In the process of executing the instructions you are currently: {current_status}.

You'll be given instructions on a route to take in the room.  **While the map is 2D the room is 3D so prepositions used in the instructions such as "above" and "below" refer to objects Z-coordinate which isn't given.**  The route will consist of one or more waypoints referenced by objects.  Using the object list, knowledge of your movement history, your current position, and spatial reasoning, follow the instructions to reach the goal.  Be sure to review your movement history to prevent unnecessary back-tracking.  Update the status with the current waypoint you're moving towards.  I will give you a list of nearby positions you can occupy without running into obstacles as x-y tuples.

Format your response as follows:
    Provide the answer as a single number.
    Please ONLY respond in the format:
        Reasoning: reason about the given answer and
        Status: moving to 'object name'
        Answer: your answer

Instructions:
   {challenge_question}  Choose the number of the tuple that is the best option to reach the goal while also following instructions:\n{movement_list}
"""
)

get_goal_position_prompt = (
"""
You are a wheeled robot in an indoor environment.  The room has been mapped to a x-y world grid.  Using your array of sensors you have detected the locations of important objects in the room.  The location of each object is given by an x-y tuple.  The object locations are defined using JSON dictionaries as follows, {{"object name": [x, y]}} where the key is the object's name and the value is a tuple composed of the object's x and y coordinates.

Here is the list of objects: {obs}.  Use the object list and spatial reasoning to answer the question.  Answer with the target object's name and center given as a JSON object as provided in the objects list.

Response Format:
    Please ONLY respond in the format:
        Reasoning: reason about the given answer and
        Answer: your answer

Question:
   Given the following instructions, what are the coordinates of the target object: {challenge_question}?
"""
)

instruction_following_prompt2 = (
"""
You are a wheeled robot in an indoor environment interacting with a human user.  The room you're in is mapped to a x-y world grid.  Using your array of sensors you have detected the locations of important objects in the room.  The location of each object is demarcated by a bounding box.  The bounding boxes are defined using JSON dictionaries as follows '{{"object name": [(min_x, min_y), (max_x, max_y)]}}' where the key is the object's name and the value is a list of two tuples.  The first tuple is the bounding minimum x and minimum y and the second tuple is the bounding box maximum x and maximum y.

The human user gives you a set of navigational instructions in English.  Convert the natural language instructions into a computer program using a language that consists of 4 commands: goto(x,y), between((x1,y1),(x2,y2)), avoid_between((x1,y1),(x2,y2), stop_at(x,y).  The 'between' and 'avoid_between' command arguments are two tuples representing bounding-box minimum x and minimum y and maximum x and maximum y, while the arguments for the "goto" and "stop_at" are points represented by tuples.  Give your answer as a JSON list of commands: '["goto(x,y)", "between((x1, y1), (x2, y2))", "avoid_between((x1, y1), (x2, y2))", "stop_at(x, y)"]'.     

Here is the list of objects: {objs}.

You are currently at position {curr_position}.

The bounding box representing the limits of the traversable area are {room_limits}.

Using the object list, your location, the room extents, and spatial reasoning convert the instructions into a program.  Explain your answer.

Format your response as follows:
   Reasoning: Explain the reasoning behind the answer, including how the target objects and path were determined.
   Answer: output should be a JSON list of commands in the order of the original instructions       
   Example Answer:
      Reasoning: explain your reasoning here
      Answer: 
      [
         "goto(x, y)",
         "between((x1, y1), (x2, y2))",
         "avoid_between"((x3, y3), (x4, y4)),
         "stop_at(x5, y5)", 
      ]

Instructions: {challenge_question}

Answer:
"""
)

instruction_following_prompt3 = (
"""
You are a wheeled robot in an indoor environment interacting with a human user.  The room you're in is mapped to a x-y world grid.  Using your array of sensors you have detected the locations of important objects in the room.  The location of the center of each object is given.  The centers are defined using JSON dictionaries as follows '{{"object name": (x, y)}}' where the key is the object's name and the value its center given as a tuple.

The human user gives you a set of navigational instructions in English.  **While the coordinates are 2D the room is 3D, so prepositions used in the instructions such as "above" and "below" refer to objects' Z-coordinate which isn't given.**  Convert the natural language instructions into a computer program using a language that consists of 4 commands: goto(x,y), between((x1,y1),(x2,y2)), avoid_between((x1,y1),(x2,y2), stop_at(x,y).  The 'between' and 'avoid_between' command arguments are two tuples representing bounding-box minimum x and minimum y and maximum x and maximum y, while the arguments for the "goto" and "stop_at" are points represented by tuples.  Give your answer as a JSON list of commands: '["goto(x,y)", "between((x1, y1), (x2, y2))", "avoid_between((x1, y1), (x2, y2))", "stop_at(x, y)"]'.

Here is the list of objects: {objs}.

You are currently at position {curr_position}.

The bounding box representing the limits of the traversable area are {room_limits}.

Using the object list, your location, the room extents, and spatial reasoning convert the instructions into a program.  Explain your answer.

Format your response as follows:
   Reasoning: Explain the reasoning behind the answer, including how the target objects and path were determined.
   Answer: output should be a JSON list of commands in the order of the original instructions
   Example Answer:
      Reasoning: explain your reasoning here
      Answer:
      [
         "goto(x, y)",
         "between((x1, y1), (x2, y2))",
         "avoid_between"((x3, y3), (x4, y4)),
         "stop_at(x5, y5)",
      ]

Instructions: {challenge_question}

Answer:
"""
)

instruction_following_prompt4 = (
"""
You are a wheeled robot in an indoor environment interacting with a human user.  The room you're in contains many objects.  The human user gives you a set of navigational instructions in English.  **While the coordinates are 2D the room is 3D, so prepositions used in the instructions such as "above" and "below" refer to objects' Z-coordinate.**  Convert the natural language instructions into a computer program using a language that consists of 3 commands: goto(object_name), between(object1_name, object2_name), and avoid_between(object1_name, object2_name).  The 'between' and 'avoid_between' command arguments are two object names while the argument for the "goto" command is one object name .  Give your answer as a JSON list of commands: '["goto(object1_name)", "between(object1_name, object2_name)", "avoid_between(object1_name, object2_name)"]'.

Convert the instructions into a program.  Explain your answer.

Format your response as follows:
   Reasoning: Explain the reasoning behind the answer, including how the target objects and path were determined.
   Answer: output should be a JSON list of commands in the order of the original instructions
   Example Answer:
      Reasoning: explain your reasoning here
      Answer:
      [
         "goto(object1_name)",
         "between(object3_name, object4_name)",
         "avoid_between"(object5_name, object6_name),
         "goto(object7_name)"
      ]

Instructions: {challenge_question}

Answer:
"""
)

cross_reference_prompt = (
"""
You are a wheeled robot in an indoor environment interacting with a human user.  You have a list of objects in the room.  The human user gives you instructions that reference the room objects.  Based off the list off objects in the room and the human user's instructions create a list of the objects the instructions reference.  Use the name given in the object list.  List each object only once.  Give your answer as a JSON list.

Object list: {room_objects}

Instructions: {instr}

Response Format:
    Please ONLY respond in the format:
        Reasoning: reason about the given answer
        Answer: ["object_name1", "object_name2", ...]
"""
)

cross_ref_coords_prompt = (
"""
You are a wheeled robot in an indoor environment interacting with a human user.  The room you're in is mapped to a x-y world grid.  Using your array of sensors you have detected the locations of important objects in the room.  The location of the center of each object is given.  The centers are defined using JSON dictionaries as follows '{{"object name": (x, y)}}' where the key is the object's name and the value its center given as a tuple.

The human user gives you a set of navigational instructions in English.  **While the coordinates are 2D the room is 3D, so prepositions used in the instructions such as "above" and "below" refer to objects' Z-coordinate which isn't given.**

Here is the list of objects in the room: {objs}.
The objects referenced in the human user's instruction correspond to objects in the above list.

You are currently at position {curr_position}.

The bounding box representing the limits of the traversable area are {room_limits}.

Objecs of interest: {targets}.

Using the object list, your location, the room extents, and spatial reasoning give the most likely coordinates of the objects of interest.  If there is not an exact match between the object of interest name and the name in the list of room objects, choose the object the user is most likely referencing.  Use the names given in the list of room objects in your answer.  Explain your answer.


Format your response as follows:
   Reasoning: Explain the reasoning behind the answer
   Answer: output should be a JSON list of dictionaries giving the objects name as found in the object list and its coordinates
   Example Answer:
      Reasoning: explain your reasoning here
      Answer:
      [
         {{"object_name1": (x1, y1)}},
         {{"object_name2": (x2, y2)}},
         .
         .
         .
      ]

Instructions: {challenge_question}

Answer:
"""
)


