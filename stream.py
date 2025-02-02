import cv2
import torch
import streamlit as st
import tempfile
import os
import time
from ultralytics import YOLO
import cv2 
from PIL import Image
import base64
import google.generativeai as genai
import io
import pandas as pd 



####################################################################################


# global housing 

# Load YOLOv8 model
model_path = "mined_model.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(model_path).to(device)
st.set_page_config(layout="wide")

left, right = st.columns(2) 

with right : 
    r1 , r2 = st.columns(2)
# Region of Interest (ROI)
x1, y1, x2, y2 = 370, 320, 480, 430
frame_counter = 0  # Frame counter for detection cooldown
total_time_detected = 0  # Total times detected



full_event_data = []

######################## REAL TIME PART HANDELING #################################

import cv2
import easyocr
import json
import numpy as np 

# Define the bounding boxes (x, y, width, height) for each label
BBOXES = {
    "health": (256, 443, 34, 25),  # Health box coordinates
    "ammo": (557, 440, 34, 32),    # Ammo box coordinates
    "ammo2": (600, 440, 15, 30)    # Ammo2 box coordinates
}

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])  # 'en' stands for English

def process_real_time_frame(frame, frame_number):
    """
    Processes a single frame, extracts text from defined bounding boxes, and returns structured data.
    """
    # Increase resolution by resizing the frame (upscale for better OCR accuracy)
    image_resized = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Scale factors for bounding boxes
    height, width, _ = image_resized.shape
    scale_x = width / frame.shape[1]
    scale_y = height / frame.shape[0]
    
    extracted_text = {}

    for label, (x, y, w, h) in BBOXES.items():
        # Adjust bounding boxes for the resized image
        x_resized = int(x * scale_x)
        y_resized = int(y * scale_y)
        w_resized = int(w * scale_x)
        h_resized = int(h * scale_y)
        
        # Crop the region of interest (ROI)
        roi = image_resized[y_resized:y_resized + h_resized, x_resized:x_resized + w_resized]
        
        # Extract text using EasyOCR
        result = reader.readtext(roi)
        
        # Store extracted text
        extracted_text[label] = ' '.join([text[1] for text in result]) if result else 'No text found'
    
    # Extract numerical values for ammo fields
    for label in ['ammo', 'ammo2']:
        if label in extracted_text:
            ammo_text = extracted_text[label]
            ammo_parts = ammo_text.split()  # Split text by spaces
            
            extracted_text[label] = ammo_parts[0] if ammo_parts and ammo_parts[0].isdigit() else np.NAN
    
    return {
        "frame_number": frame_number,
        "data": extracted_text
    }






#######################GEN AI ######################################################

genai.configure(api_key="AIzaSyBDMPacCxo-ODoeXAQ6dg5PFZXUs0-I1UU")





def gen_report(frame) : 

  _, image_bytes = cv2.imencode('.png', frame)

  image_base64 = base64.b64encode(image_bytes).decode("utf-8")

  # Initialize the model
  model = genai.GenerativeModel("gemini-1.5-flash")

  # Generate response
  response = model.generate_content(
      [{'mime_type': 'image/png', 'data': image_base64}, """
      
      the following is a scropped screen shot of a valorant game 
                  the information in game is in forllowing format : 

                  
                  | player 1 name | gun symbol | symbol of head shot (if it is presnt) | player 2 name | 

                  the information will be avilable on the right most corner of teh image 
                  also your first priority is to locate the kill notification related to me 

                  also on above there is an information displayed of the following 

                  how many rounds team green won 
                  and how many rounds team red won 
                  the current til in current round 
                  and map information  for eg ( atacked side span , written above map)


                  with a wepon 
                  Also check for headshot or simple kill in given image 
                  there will be a symbol in middle with headshot written in it if it is a head shot 

                  
                  you need to give the following output : 

                
                
                  for example : 
                  {
                      player_in_left : "chamber" ,
                      player_in_right : "me" , 
                      weapon_type : "vandal" ,
                      time : " the curent time " , 
                    }

                  just think by your self do not use any tool 

                  output : 
                  {
                  
                  

                  "game_info" : [
                    {
                      player_in_right : "name of player" ,
                      player_in_left : "name of player" , 
                      weapon_type : "valorant gun type  , if you dont correctly guess the gun type just give any gun type that is present in valorant "
      
                  
                    } , 
                    ..... 
                  ] , {

                      total_rounds_won_by_red : "2" , 
                      total_rounds_won_by_green : "4" , 
                      map_info  : "given above map"   , 
                      current_time  : "time" , 
                      headshot : "true / false " , 
                      summary : {
                       summary_title : "the title you tink is apt" , 
                       full_summary : "also describe the scenario happened in gameplay,
                      i am providing one of the frame of video when any kill is detected,
                      explain how complex was this shot , 
                      The complexity of the shot would be determine if the kill is headshot or 
                      killed by peeking through wall or kill through wall etc.. are the factors which determine the complexity of shot.
                      explain the complexity of the shot 
                      write it in consise manner 2-3 lines." 
                        }
                  } 

                  }

       
           note :  the following is an example of a good summary 
        { 
         summary_title : "a relatively simple kill" , 
         full_summary : "Reyna was likely visible to Sage, and the kill wasn't achieved through a particularly difficult angle or technique.  The lack of a headshot symbol suggests a standard body shot. The shot's complexity is low."
       
       }


        
      """] , 
          generation_config=genai.GenerationConfig(
          response_mime_type="application/json"
      )
      
  )

  import json

  # Extract text part correctly
  text_data = response.candidates[0].content.parts[0].text

  # Convert JSON string to dictionary
  parsed_data = json.loads(text_data)

  # Print extracted information
  return parsed_data


















########################### YOLO #################################################


def page1() : 
   def process_video(video_path):
      global frame_counter, total_time_detected  , to_write # Persistent across frames


      
      cap = cv2.VideoCapture(video_path)
      fps = cap.get(cv2.CAP_PROP_FPS)
      cooldown_frames = int(3 * fps)  # Convert 3 seconds to frames
      print(cooldown_frames)
      stframe = st.empty()

      runner = 0 

      with right : 
         
         d_container = st.empty()
         r1 , r2 = st.columns(2)
         with r1 : 
               d1_container = st.empty()
               g1_container = st.empty()
         with r2 : 
               d2_container = st.empty()
               g2_container = st.empty()

      # Create a container for displaying detection count

      real_time_data = []


      while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
         # Frame is empty, save full_event_data to a JSON file
               with open('data.json', 'w') as json_file:
                  json.dump(full_event_data, json_file)
               break
            frame = cv2.resize(frame, dsize=(852 , 480))


            store_frame_data = "na"
            store_event_report = "na"
            store_timestamp = "na"


            if not ret:
                  break

            results = model(frame)
            detected = False  

            frame_data = process_real_time_frame(frame , runner)
            store_frame_data = frame_data
            runner += 1 

            real_time_data.append(frame_data)

            df = pd.DataFrame([
            {
         "frame_number": item["frame_number"],
         "health": float(item["data"]["health"]) if str(item["data"]["health"]).isdigit() else np.nan,
         "ammo": float(item["data"]["ammo"]) if str(item["data"]["ammo"]).isdigit() else np.nan,
         "ammo2": float(item["data"]["ammo2"]) if str(item["data"]["ammo2"]).isdigit() else np.nan
               }
               for item in real_time_data
            ])

            # Replace 0 with NaN if 0 should be treated as missing data
            df.replace(0, np.nan, inplace=True)

            # Fill NaN values first using forward and backward fill
            df.fillna(method='bfill', inplace=True)  # Fill using next available value
            df.fillna(method='ffill', inplace=True)  # Fill remaining NaNs using previous values

            # Convert float columns safely to int after filling NaNs
            df = df.astype({"health": int, "ammo": int, "ammo2": int})
                     
            with right : 
               
               r1 , r2 = st.columns(2)

               with r1 : 
                  g1_container.line_chart(df.set_index("frame_number")["health"] , x_label="health")
               with r2 : 
                  g2_container.line_chart(df.set_index("frame_number")[["ammo", "ammo2"]])




            for result in results:
                  for box in result.boxes:
                     x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                     confidence = float(box.conf[0])
                     center_x = (x_min + x_max) // 2
                     center_y = (y_min + y_max) // 2

                     if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                        detected = True
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
                        text = f"{confidence:.2f}"
                        cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, (0, 255, 0), 2, cv2.LINE_AA)

            # Detection logic using frames instead of time
            if detected:
               if frame_counter == 0 or frame_counter >= cooldown_frames:
                  frame_counter = 1  # Reset counter after detection

               else:
                  if frame_counter == 4 : 
                     
                     with right:
                        d_container.empty()
                        d1_container.empty()
                        d2_container.empty()



                        r1 , r2 = st.columns(2)
                        total_time_detected += 1
                        # Clear previous detection count
                        right.empty()
                        # Get the report data
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        data = gen_report(frame_rgb)
                        print(data)
                        # Filter game info to include only entries with "Me" or "me" in player names (case-insensitive)
                        game_info_filtered = [
                           game for game in data['game_info']
                           if 'me' in game['player_in_right'].lower() or 'me' in game['player_in_left'].lower()
                        ]
                        
                        if len(game_info_filtered) == 0 :
                           game_info_filtered = [data['game_info'][0]]

                        d_container.markdown(f"""
                           ## {data['summary']['summary_title']}
                           {data['summary']['full_summary']}
                           """)

                        # Display the formatted markdown text
                        with r1 : 
                           d1_container.markdown(f"""
                              ## Player Info
                                       
                              - *Player in Right*: {game_info_filtered[0]['player_in_right']}
                              - *Player in Left*: {game_info_filtered[0]['player_in_left']}
                              - *Weapon Type*: {game_info_filtered[0]['weapon_type']}

                              ## Round Info
                              - *Total Rounds Won by Red Team*: {data['total_rounds_won_by_red']}
                              - *Total Rounds Won by green Team*: {data['total_rounds_won_by_green']}
                              """)
                           
                        with r2 : 
                           d2_container.markdown(f"""
                              ## Map Info
                              - *Map*: {data['map_info']}

                              ## Current Time
                              - *Time*: {data['current_time']}

                              ## Headshot Status
                              - *Headshot*: {data['headshot']}
                           """)
                           
                        store_event_report = data
                           

                        full_event_data.append({
                           
                           'frame_data' : store_frame_data , 
                           'event_report' : store_event_report , 
                           'event_timestamp' : runner/fps
                        })

                        with open('data.json', 'w') as json_file:
                           json.dump(full_event_data, json_file, indent=4)
                        
                  frame_counter += 1  # Continue counting frames
            else:
               if frame_counter > 0:
                  frame_counter += 1  # Continue counting frames if within cooldown


            # Convert BGR to RGB for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB")




            # Maintain FPS
            time.sleep(1 / fps)

   

   # Streamlit UI

   with left : 
      st.title("YOLOv8 Video Object Detection")
      uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

      if uploaded_file is not None:
         video_path = "video.mp4"  # Store the video in the root directory

         with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

         st.write("Processing video...")
         process_video(video_path)





def page2() : 
   def process_game_data(data):
      markdown_text = ""
      
      for entry in data:
         frame_data = entry['frame_data']
         event_report = entry['event_report']
         
         frame_number = frame_data['frame_number']
         health = frame_data['data']['health']
         ammo = frame_data['data']['ammo']
         ammo2 = frame_data['data']['ammo2']
         
         game_info = event_report['game_info'][0]
         player_in_right = game_info['player_in_right']
         player_in_left = game_info['player_in_left']
         weapon_type = game_info['weapon_type']
         
         total_rounds_won_by_red = event_report['total_rounds_won_by_red']
         total_rounds_won_by_green = event_report['total_rounds_won_by_green']
         map_info = event_report['map_info']
         current_time = event_report['current_time']
         headshot = event_report['headshot']
         
         summary_title = event_report['summary']['summary_title']
         full_summary = event_report['summary']['full_summary']
         
         event_timestamp = entry['event_timestamp']
         
         
         markdown_text += f"**Event Timestamp:** {event_timestamp}\n\n"
         markdown_text += f"**Health:** {health} | **Ammo:** {ammo} | **Ammo2:** {ammo2}\n\n"
         markdown_text += f"**Game Information:**\n"
         markdown_text += f"- **Player (Right):** {player_in_right}\n"
         markdown_text += f"- **Player (Left):** {player_in_left}\n"
         markdown_text += f"- **Weapon Type:** {weapon_type}\n\n"
         markdown_text += f"**Round Statistics:**\n"
         markdown_text += f"- **Total Rounds Won by Red:** {total_rounds_won_by_red}\n"
         markdown_text += f"- **Total Rounds Won by Green:** {total_rounds_won_by_green}\n"
         markdown_text += f"- **Map:** {map_info}\n"
         markdown_text += f"- **Current Time:** {current_time}\n"
         markdown_text += f"- **Headshot:** {headshot}\n\n"
         markdown_text += f"**Summary:**\n"
         markdown_text += f"- **Title:** {summary_title}\n"
         markdown_text += f"- **Details:** {full_summary}\n\n"
         markdown_text += f"### Frame Number: {frame_number}\n"
         markdown_text += "---\n\n"
      
      return markdown_text


   with open('data.json', 'r') as f:
      data = json.load(f)

   context = process_game_data(data)


   from langchain_groq import ChatGroq


   llm = ChatGroq(
      model="llama-3.3-70b-versatile",
      temperature=0,
      max_tokens=None,
      timeout=None,
      max_retries=2,
      api_key="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"
               
   )




   prompt = f""" 

   given the following context : 


   {context} 


   context is the repoted values at key events 
   based on that values give the following answers and note : 
   try to answer as much as you can with the help of the given data
   and in answer do not indicate the lack of data


   comment about the following parameters 

   Player Statistics :
   --> Calculate total kills and headshot percentage.
   --> health / damage analysis
   2. Match Timeline :
   --> Provide a chronological summary of key events.
   --> Highlight round wins and notable player contributions.
   3. Weapon Analysis :
   --> Identify weapons used and calculate their performance metrics (e.g., accuracy, kills).
   --> Track time spent using each weapon or switching between them.
   4. Player Behavior Analysis
   --> Assess the aggressiveness level of the player based on time spent engaging opponents.





   """



   answer = llm.invoke(prompt)


   st.title( " AI 🤖 match summary")

   st.markdown(answer.content)

   

def page3():
      
   def ask_game(query) : 

      def process_game_data(data):
         markdown_text = ""
         
         for entry in data:
            frame_data = entry['frame_data']
            event_report = entry['event_report']
            
            frame_number = frame_data['frame_number']
            health = frame_data['data']['health']
            ammo = frame_data['data']['ammo']
            ammo2 = frame_data['data']['ammo2']
            
            game_info = event_report['game_info'][0]
            player_in_right = game_info['player_in_right']
            player_in_left = game_info['player_in_left']
            weapon_type = game_info['weapon_type']
            
            total_rounds_won_by_red = event_report['total_rounds_won_by_red']
            total_rounds_won_by_green = event_report['total_rounds_won_by_green']
            map_info = event_report['map_info']
            current_time = event_report['current_time']
            headshot = event_report['headshot']
            
            summary_title = event_report['summary']['summary_title']
            full_summary = event_report['summary']['full_summary']
            
            event_timestamp = entry['event_timestamp']
            
            
            markdown_text += f"**Event Timestamp:** {event_timestamp}\n\n"
            markdown_text += f"**Health:** {health} | **Ammo:** {ammo} | **Ammo2:** {ammo2}\n\n"
            markdown_text += f"**Game Information:**\n"
            markdown_text += f"- **Player (Right):** {player_in_right}\n"
            markdown_text += f"- **Player (Left):** {player_in_left}\n"
            markdown_text += f"- **Weapon Type:** {weapon_type}\n\n"
            markdown_text += f"**Round Statistics:**\n"
            markdown_text += f"- **Total Rounds Won by Red:** {total_rounds_won_by_red}\n"
            markdown_text += f"- **Total Rounds Won by Green:** {total_rounds_won_by_green}\n"
            markdown_text += f"- **Map:** {map_info}\n"
            markdown_text += f"- **Current Time:** {current_time}\n"
            markdown_text += f"- **Headshot:** {headshot}\n\n"
            markdown_text += f"**Summary:**\n"
            markdown_text += f"- **Title:** {summary_title}\n"
            markdown_text += f"- **Details:** {full_summary}\n\n"
            markdown_text += f"### Frame Number: {frame_number}\n"
            markdown_text += "---\n\n"
         
         return markdown_text


      with open('data.json', 'r') as f:
         data = json.load(f)

      context = process_game_data(data)


      from langchain_groq import ChatGroq


      llm = ChatGroq(
         model="llama-3.3-70b-versatile",
         temperature=0,
         max_tokens=None,
         timeout=None,
         max_retries=2,
         api_key="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"
                  
      )




      prompt = f""" 

      given the following context : 


      {context} 


      context is the repoted values at key events 
      based on that values give the following answers and note : 
      try to answer as much as you can with the help of the given data
      and in answer do not indicate the lack of data


      answer the following query 

      {query}

      """



      answer = llm.invoke(prompt)

      return answer.content
   
   st.title("Ask Ai 🤖 about the game")
   
   query = st.text_input("Enter your question")
   st.markdown(ask_game(query))

   
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.fx.all import fadein, fadeout

def merge_videos_with_transitions(input_path, output_path, timestamps):
    # Load the video
    video = VideoFileClip(input_path)
    
    video_duration = video.duration  # Get the total duration of the video
    
    clips = []
    for i, timestamp in enumerate(timestamps):
        # Define the start and end times for the clip
        start_time = max(0, timestamp - 2)  # Ensure the start time is not negative
        end_time = min(timestamp + 2, video_duration)  # Ensure end time does not exceed video duration
        
        # Get the segment of the video around the timestamp
        clip = video.subclip(start_time, end_time)
        
        # Apply fade-in and fade-out transitions to each clip
        if i == 0:  # Fade-in for the first clip
            clip = fadein(clip, 1)  # 1-second fade-in
        else:  # Fade-in for the subsequent clips
            clip = fadein(clip, 1)
        
        if i == len(timestamps) - 1:  # Fade-out for the last clip
            clip = fadeout(clip, 1)  # 1-second fade-out
        else:  # Fade-out for the intermediate clips
            clip = fadeout(clip, 1)
        
        clips.append(clip)
    
    # Concatenate the clips with transitions
    final_clip = concatenate_videoclips(clips, method="compose")
    
    # Write the final video to the output path
    final_clip.write_videofile(output_path, codec="libx264")

def make_highlight() : 
   def process_game_data(data):
      markdown_text = ""
      
      for entry in data:
         frame_data = entry['frame_data']
         event_report = entry['event_report']
         
         frame_number = frame_data['frame_number']
         health = frame_data['data']['health']
         ammo = frame_data['data']['ammo']
         ammo2 = frame_data['data']['ammo2']
         
         game_info = event_report['game_info'][0]
         player_in_right = game_info['player_in_right']
         player_in_left = game_info['player_in_left']
         weapon_type = game_info['weapon_type']
         
         total_rounds_won_by_red = event_report['total_rounds_won_by_red']
         total_rounds_won_by_green = event_report['total_rounds_won_by_green']
         map_info = event_report['map_info']
         current_time = event_report['current_time']
         headshot = event_report['headshot']
         
         summary_title = event_report['summary']['summary_title']
         full_summary = event_report['summary']['full_summary']
         
         event_timestamp = entry['event_timestamp']
         
         
         markdown_text += f"**Event Timestamp:** {event_timestamp}\n\n"
         markdown_text += f"**Health:** {health} | **Ammo:** {ammo} | **Ammo2:** {ammo2}\n\n"
         markdown_text += f"**Game Information:**\n"
         markdown_text += f"- **Player (Right):** {player_in_right}\n"
         markdown_text += f"- **Player (Left):** {player_in_left}\n"
         markdown_text += f"- **Weapon Type:** {weapon_type}\n\n"
         markdown_text += f"**Round Statistics:**\n"
         markdown_text += f"- **Total Rounds Won by Red:** {total_rounds_won_by_red}\n"
         markdown_text += f"- **Total Rounds Won by Green:** {total_rounds_won_by_green}\n"
         markdown_text += f"- **Map:** {map_info}\n"
         markdown_text += f"- **Current Time:** {current_time}\n"
         markdown_text += f"- **Headshot:** {headshot}\n\n"
         markdown_text += f"**Summary:**\n"
         markdown_text += f"- **Title:** {summary_title}\n"
         markdown_text += f"- **Details:** {full_summary}\n\n"
         markdown_text += f"### Frame Number: {frame_number}\n"
         markdown_text += "---\n\n"
      
      return markdown_text


   with open('data.json', 'r') as f:
      data = json.load(f)

   context = process_game_data(data)

   import google.generativeai as genai

   genai.configure(api_key="AIzaSyC9KkbgmUDIB8BbiaKDmjrxTVI1omRh-TQ")

   model = genai.GenerativeModel("gemini-1.5-flash")





   prompt = """ 

   given the following context : 

   """ + f"""

   {context} 

   """ + """

   from the following information give top 5 time stamps (if data is less than 5 give all)
   on which highlights can be made , due to exciting summary 

   answer in the following json fromat 

   {
      "highlights": [
         "time_stamp_1" ,
         "time_stamp_2" ,
         "time_stamp_3" ,
         "time_stamp_4" ,
         "time_stamp_5" 
      ]
   }

   eg : {

   "highlights": [
         10 ,
         5 , 
         6 , 
         18 , 
         25
      ]

   }



   """

   model = genai.GenerativeModel("gemini-1.5-pro-latest")
   result = model.generate_content(
      prompt,
      generation_config=genai.GenerationConfig(
         response_mime_type="application/json"
      ),
   )

   text_data = result.candidates[0].content.parts[0].text

   # Convert JSON string to dictionary
   parsed_data = json.loads(text_data)

   timestamps = [float(time) for time in parsed_data['highlights']]
   merge_videos_with_transitions("video.mp4", "output.mp4", timestamps)

def page4():
    st.title("Your Video Highlights")
    make_highlight()
    st.video("output.mp4")




############################################## Hologram ######################################






import cv2
import os

# Extract frames from video in 1080p within a specific time range
def extract_frames(video_path, output_folder, fps=30, resolution=(1920, 1080), start_sec=0, end_sec=None):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / original_fps  # Total video duration in seconds
    interval = int(original_fps / fps)
    
    # Calculate start and end frames
    start_frame = int(start_sec * original_fps)
    end_frame = int(end_sec * original_fps) if end_sec else total_frames

    # Ensure start and end are within the valid range
    start_frame = max(0, min(start_frame, total_frames))
    end_frame = max(start_frame, min(end_frame, total_frames))
    
    # Set the start position in the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_id = start_frame
    frame_count = 0
    while cap.isOpened() and frame_id < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frames based on the interval
        if frame_id % interval == 0:
            # Resize frame to 1080p if necessary
            if frame.shape[1] != resolution[0] or frame.shape[0] != resolution[1]:
                frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)
            output_path = f"{output_folder}/frame_{frame_count:05d}.jpg"
            cv2.imwrite(output_path, frame)
            frame_count += 1
        
        frame_id += 1

    cap.release()
    print(f"Frames extracted from {start_sec}s to {end_sec}s and saved to {output_folder}.")

# Example usage




import cv2
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import os
from gameplay_analysis import GameplayAnalysis

class BatchFrameProcessor:
    def __init__(self, input_folder, output_folder, full_map_path, yolo_model_path):
        """
        Initialize the batch frame processor
        
        Args:
            input_folder (str): Path to folder containing input frames
            output_folder (str): Path to store processed frames
            full_map_path (str): Path to the full map image
            yolo_model_path (str): Path to YOLO model weights
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize the analyzer
        self.analyzer = GameplayAnalysis(full_map_path, yolo_model_path)
        
    def process_frames(self):
        """Process all frames in the input folder"""
        # Get all frame files
        frame_files = sorted(list(self.input_folder.glob('*.jpg'))) + \
                     sorted(list(self.input_folder.glob('*.png')))
        
        if not frame_files:
            raise ValueError(f"No frames found in {self.input_folder}")
            
        self.logger.info(f"Found {len(frame_files)} frames to process")
        
        # Process each frame
        for frame_path in tqdm(frame_files, desc="Processing frames"):
            try:
                # Read frame
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    self.logger.warning(f"Failed to read frame: {frame_path}")
                    continue
                    
                # Process frame
                result = self.analyzer.process_frame(frame)
                if result is None:
                    self.logger.warning(f"Failed to process frame: {frame_path}")
                    continue
                    
                # Save processed frame
                output_path = self.output_folder / frame_path.name
                cv2.imwrite(str(output_path), result)
                
            except Exception as e:
                self.logger.error(f"Error processing frame {frame_path}: {str(e)}")
                
    def create_video(self, output_video_path, fps=30):
        """Create video from processed frames"""
        try:
            # Get all processed frames
            processed_frames = sorted(list(self.output_folder.glob('*.jpg'))) + \
                             sorted(list(self.output_folder.glob('*.png')))
            
            if not processed_frames:
                raise ValueError("No processed frames found")
                
            # Read first frame to get dimensions
            first_frame = cv2.imread(str(processed_frames[0]))
            height, width = first_frame.shape[:2]
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_video_path,
                fourcc,
                fps,
                (width, height)
            )
            
            # Write frames to video
            self.logger.info("Creating video...")
            for frame_path in tqdm(processed_frames, desc="Creating video"):
                frame = cv2.imread(str(frame_path))
                video_writer.write(frame)
                
            video_writer.release()
            self.logger.info(f"Video saved to {output_video_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating video: {str(e)}")
            raise

def runner1():
    # Paths configuration
    INPUT_FOLDER = "frames_1080p"
    OUTPUT_FOLDER = "processed_frames"
    FULL_MAP_PATH = "full_map.png"
    YOLO_MODEL_PATH = "player_tracker.pt"
    OUTPUT_VIDEO_PATH = "output_video.mp4"
    
    try:
        # Initialize processor
        processor = BatchFrameProcessor(
            input_folder=INPUT_FOLDER,
            output_folder=OUTPUT_FOLDER,
            full_map_path=FULL_MAP_PATH,
            yolo_model_path=YOLO_MODEL_PATH
        )
        
        # Process all frames
        processor.process_frames()
        
        # Create video from processed frames
        processor.create_video(OUTPUT_VIDEO_PATH)
        
    except Exception as e:
        logging.error(f"Batch processing failed: {str(e)}")





    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from glob import glob

    # Path to folder containing map images with red dots
    IMAGE_FOLDER = "processed_frames"
    BASE_MAP_PATH = "full_map.png"  # A clean base map without red dots

    # Load base map
    base_map = cv2.imread(BASE_MAP_PATH)
    base_map = cv2.cvtColor(base_map, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting

    # Get all image paths
    image_paths = glob(os.path.join(IMAGE_FOLDER, "*.jpg"))  # Adjust extension if needed

    # Heatmap storage
    heatmap = np.zeros((base_map.shape[0], base_map.shape[1]), dtype=np.float32)

    # Function to detect red dots and extract player positions
    def extract_positions(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define HSV range for detecting red color
        lower_red1 = np.array([0, 100, 100])   # Lower bound for red
        upper_red1 = np.array([10, 255, 255])  # Upper bound for red
        lower_red2 = np.array([160, 100, 100]) # Second range (red wraps around in HSV)
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Find contours of red dots
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        positions = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 5:  # Filter small noise
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])  # X centroid
                    cy = int(M["m01"] / M["m00"])  # Y centroid
                    positions.append((cx, cy))
        
        return positions

    # Process each image
    for img_path in image_paths:
        img = cv2.imread(img_path)
        positions = extract_positions(img)
        
        # Add player positions to heatmap
        for (x, y) in positions:
            heatmap[y, x] += 1  # Increment density at player positions

    # Normalize heatmap
    heatmap = cv2.GaussianBlur(heatmap, (101, 101), 0)  # Smooth the heatmap
    heatmap = heatmap / heatmap.max()  # Normalize values between 0 and 1

    # Apply colormap
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Blend heatmap with base map
    overlay = cv2.addWeighted(base_map, 0.6, heatmap_color, 0.4, 0)

    cv2.imwrite("player_heatmap.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))




def deleter() : 
    # deleting folder "frames_1080p"
    import shutil
    try:
        shutil.rmtree('frames_1080p')
    except Exception as e:
        print(f"Error while deleting {e}")

    # deleting folder "processed_frames"
    try:
        shutil.rmtree('processed_frames')
    except Exception as e:
        print(f"Error while deleting {e}")


##########################################################################################################


import time 


def page5():
   st.title("Upload Video and Map")

   # File uploaders stacked
   video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
   map_file = st.file_uploader("Upload a map image", type=["png", "jpg", "jpeg"])

   # Save files if uploaded
   if video_file is not None:
      with open("video2.mp4", "wb") as f:
         f.write(video_file.getbuffer())

   if map_file is not None:
      with open("full_map.png", "wb") as f:
         f.write(map_file.getbuffer())

   if map_file is not None and video_file is not None:
      # Run processing functions
      extract_frames("video2.mp4", "frames_1080p", fps=10, resolution=(1920, 1080))
      runner1()
      deleter()
      time.sleep(2)

      # Display results after processing
      st.write("## Processed Outputs")
      
      st.image("player_heatmap.jpg", caption="Player Heatmap")
      st.write("Heatmap Visualization")
      
      # Download link for processed video
      with open("output_video.mp4", "rb") as file:
         st.download_button(label="Download Processed Video", data=file, file_name="output_video.mp4", mime="video/mp4")




#############################################################################################################


page_names_to_funcs = {
    "realtime infernce": page1 ,
    "over all statistics" : page2 , 
    "user chat bot" : page3 , 
    "user high light" : page4 , 
    "heatmap generation" : page5
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()