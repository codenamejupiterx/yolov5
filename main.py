#SOURCE: https://github.com/ultralytics/yolov5
#helpful link: https://github.com/sondrion/SMOKE_CLASS_DETECTION
#train custom data link:  https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
#training command (make sure you use python3 not just python and add --workers 0 to the end example below:
# #python3 train.py --img 640 --batch 16 --epochs 300 --data /Users/benjaminhall/Documents/GitHub/yolov5/data/head_detect_dataset/data.yaml --weights yolov5s.pt --workers 0)
#-------------------------------------------------------------------------------------------------------------------------

import torch
import os
import json
import datetime
import time
import shutil
from os.path import exists

with open("facdet_config.json", "r") as json_file:
    json_load = json.load(json_file)

EVENT_STREAM = []
WEIGHTS_PATH = json_load["weights_path"]
IMAGE_PATH = json_load["image_path"]
VID_PATH = json_load["vid_path"]
EVENT_STREAM_PATH = json_load["event_stream_path"]



# initialize event stream file
time_that_facdet_event_stream_began = datetime.datetime.utcnow()
time_that_individual_event_begins = ""
time_that_individual_event_ends = ""




# Getting Timestamp info
# https://www.geeksforgeeks.org/get-current-timestamp-using-python/
# beginning timestamp
stream_process = {
    "ProcessorType": "FireAndSmokeDetector",
    "StreamStart": str(time_that_facdet_event_stream_began),
    "Events": [],
}

with open(EVENT_STREAM_PATH, "w") as outfile:
    json.dump(stream_process, outfile, indent=4)

def delete_former_event_stream(EVENT_STREAM_PATH: str) -> None:
    """Will take the path to the "fsd_event_stream" file in the project directory.
    Deletes the previous "fsd_event_stream" file.
    Args:
        EVENT_STREAM_PATH (str): Path to the "fsd_event_stream" json file in the directory.
    Return:
        -NONE-
    """
    if exists(EVENT_STREAM_PATH):
        os.remove(EVENT_STREAM_PATH)
        time.sleep(10)

def event_stream_writer(starting_time, ending_time) -> None:
    """Will take a pandas dataframe holding results/stats from an individual video frame.
    It uses the results/stats to write out the event stream for the module.
    Args:
        result_dataframe (pd.DataFrame): Pandas data-structure that holds information about the inferences found in the current frame.
        preview_list_dict (list): Preview list of 2 dictionaries, 1. Type = "Image", 2 Path= "...path to that image..."
        starting_time (pd.DataFrame): Event starting time value held in a Pandas data-structure.
        ending_time (pd.DataFrame): Event ending time value held in a Pandas data-structure.
        smoke_event_or_fire_event (str): String vaules of either "fire" or "smoke".
        starting_frame_nums (int): Frame number of a starting event.
        ending_frame_nums (int): Frame number of a ending event.
    Return:
          -NONE-
    """
    event_details = []
    preview_list = []
    fire_frame_num_check = {}

   

       

  

    start = starting_time - time_that_facdet_event_stream_began
    end = ending_time - time_that_facdet_event_stream_began

    
    event = {
        "Type": "Face-Detect-Event",
        "Start": start.total_seconds(),
        "End": end.total_seconds(),
        "Attribute": event_details,
        "Previews": preview_list,
    }
    

    # appending to EVENT list in Json object json file
    stream_process["Events"].append(event)

    # writing the new list element to the file
    with open(EVENT_STREAM_PATH, "w") as outfile:
        json.dump(stream_process, outfile, indent=4)




def main():

    global time_that_individual_event_begins
    global time_that_individual_event_ends

    #facial_recognition
    model = torch.hub.load('ultralytics/yolov5', 'custom',WEIGHTS_PATH, force_reload=True)

    #----------for single image---------------------#

    # time when individual event begin(ending time is in the event steam writer function)
    
    time_that_individual_event_begins = datetime.datetime.utcnow()

    #Inference
    results = model(IMAGE_PATH)

    # Results
    results.show()
    results.save()
    results.print()
    print(results.pandas().xyxy[0])  # im predictions (pandas)
    time_that_individual_event_ends = datetime.datetime.utcnow()
    # Sending info to the event sream writer
    event_stream_writer(time_that_individual_event_begins, time_that_individual_event_ends) 
    #-----------------------------------------------#



    #------------for video frames--------------------#

    #for i in range(len(os.listdir(vid_path))):
    for i in range(407,930):

        # Images
        #img = path_to_front_of_head_pic  # or file, Path, PIL, OpenCV, numpy, list
        #img = pic_list[i]
        video_frame = VID_PATH+str(i)+".jpg"

        
        time_that_individual_event_begins = datetime.datetime.utcnow()

        # Inference
        results = model(video_frame)

        # Results
        #results.show()
        results.save()
        results.print()
        print(results.pandas().xyxy[0])  # im predictions (pandas)
        time_that_individual_event_ends = datetime.datetime.utcnow()

    # Sending info to the event sream writer
        event_stream_writer(time_that_individual_event_begins, time_that_individual_event_ends)    
    #-----------------------------------------------------------#

        
if __name__ == "__main__":
    delete_former_event_stream(EVENT_STREAM_PATH)
    main()
