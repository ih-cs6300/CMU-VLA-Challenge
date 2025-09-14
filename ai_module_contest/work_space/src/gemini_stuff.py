import cv2
import base64
# from google import genai  # requires python > 3.8
# from google.genai import types  # requires python > 3.8
import os
import numpy as np
import json
import time
import argparse
from copy import deepcopy
import re
import requests

def url_request_gemini(query):
    api_key=os.getenv('GAK')
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}'
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": query
                    }
                ]
            },
        ],
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def post_process_numerical(gemini_output):
   match = re.search(r"Answer:(.*)", gemini_output, re.DOTALL)
   result = int(match.group(1))
   json_obj = {'answer': result}
   return json_obj

def post_process_obj_extraction(gemini_output):
   if ('json' in gemini_output.lower()):
      json_obj = json.loads(gemini_output[8:-3], strict=False)
   else:
      json_obj = json.loads(gemini_output, strict=False)
   return json_obj

def post_process_obj_reference(gemini_output):
   match = re.search(r"Answer:(.*)", gemini_output, re.DOTALL)
   temp_out = match.group(1)
   if ('json' in temp_out.lower()):
      json_obj = json.loads(temp_out[8:-3])
   else:
      json_obj = json.loads(temp_out)
   return json_obj

def post_process_instruction_following(gemini_output):
   match_ans = re.search(r"Answer:(.*)", gemini_output, re.DOTALL)
   json_obj = json.loads(match_ans.group(1).lower().replace('json', '').replace('```', ''))
   return json_obj

def post_process_cross_reference(gemini_output):
   match_ans = re.search(r"Answer:(.*)", gemini_output, re.DOTALL)
   json_obj = json.loads(match_ans.group(1).lower().replace('json', '').replace('```', ''))
   return json_obj


def ask_gemini(prompt, chat_history, statement_type): 
   # The client gets the API key from the environment variable `GEMINI_API_KEY`.   
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
   elif (statement_type == 'cross-reference'):
      post_process_fxn = post_process_cross_reference
   else:
      post_process_fxn = None

   while True:
      if count > 5:
         raise Exception("Failed to get response from Gemini API")
      try:
         # The client gets the API key from the environment variable `GEMINI_API_KEY`.
         if chat_history is not None:
            response = url_request_gemini(chat_history)
         else:
            response = url_request_gemini(prompt) # gemini-2.5-flash-lite, gemini-2.5-flash
         json_obj = post_process_fxn(response)
         if (len(json_obj) > 0):
            break
      except Exception as e:
         print(e)
         print('Generic expection')
         count += 1
   if len(json_obj) == 0:
      return None
   return json_obj, response
