import cv2
import base64
from openai import OpenAI
import os
import numpy as np
import json
#import dotenv
import time
import argparse
import openai
from copy import deepcopy
import re
import httpx

def post_process_numerical(openai_output):
   match = re.search(r"Answer: (.*)", openai_output)
   result = int(match.group(1))
   json_obj = {'answer': result}
   return json_obj

def post_process_obj_extraction(openai_output):
   json_obj = json.loads(openai_output, strict=False)
   return json_obj

def post_process_obj_reference(openai_output):
   match = re.search(r"Answer: (.*)", openai_output)
   json_obj = json.loads(match.group(1))
   return json_obj

def post_process_instruction_following(openai_output):
   match_status = re.search(r"Status: (.*)", openai_output)
   status = match_status.group(1)
   match_ans = re.search(r"Answer: (.*)", openai_output)
   result = int(match_ans.group(1))
   json_obj = {'answer': result, 'status': status}
   return json_obj


def ask_openai(prompt, statement_type):
   query_obj = QueryVLM('gemini')
   #frame = image_resize_for_vlm(frame)
   PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
            ]
        },
   ]
    
   client_gpt4v = OpenAI(
      api_key=os.getenv('OPENAI_KEY'),
      timeout=httpx.Timeout(60.0, read=60.0, write=10.0, connect=3.0)
   )
   params = {
            "model": "gpt-4.1",  # gpt-4o-mini  gpt-4o   gpt-4.1 gpt-4.1-nano  o4-mini
            "messages": PROMPT_MESSAGES,
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.5,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
   }
   
   count = 0

   if (statement_type == 'object-extraction'):
      post_process_fxn = post_process_obj_extraction
   elif (statement_type == 'numerical'):
      post_process_fxn = post_process_numerical
   elif (statement_type == 'object-reference'):
      post_process_fxn = post_process_obj_reference
   elif (statement_type == 'target-coordinates'):
      post_process_fxn = post_process_obj_reference
   elif (statement_type == 'instruction-following'):
      post_process_fxn = post_process_instruction_following
   else:
      post_process_fxn = None

   while True:
      if count > 5:
         raise Exception("Failed to get response from Azure OpenAI")
      try:
         result = client_gpt4v.chat.completions.create(**params)

         # check result message has correct form; if not redo
         result_msg = result.choices[0].message.content
         json_obj = post_process_fxn(result_msg)
         if (len(json_obj) > 0):
            break
      except openai.BadRequestError as e:
         print(e)
         print('Bad Request error.')
         return None, None
      except openai.RateLimitError as e:
         print(e)
         print('Rate Limit. Waiting for 5 seconds...')
         time.sleep(5)
         count += 1
      except openai.APIStatusError as e:
         print(e)
         print('APIStatusError. Waiting for 1 second...')
         time.sleep(1)
         count += 1
      except Exception as e:
         print(e)
         print('Generic expection')
         count += 1
   if len(json_obj) == 0:
      return None
   return json_obj, result.choices[0].message.content
