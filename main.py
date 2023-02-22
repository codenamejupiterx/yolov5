# SOURCE: https://github.com/ultralytics/yolov5
# helpful link: https://github.com/sondrion/SMOKE_CLASS_DETECTION
# train custom data link:  https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
# training command (make sure you use python3 not just python and add --workers 0 to the end example below:
# #python3 train.py --img 640 --batch 16 --epochs 300 --data /Users/benjaminhall/Documents/GitHub/yolov5/data/head_detect_dataset/data.yaml --weights yolov5s.pt --workers 0)
# -------------------------------------------------------------------------------------------------------------------------

import torch
import os
import json
import pandas as pd
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
OUTPUT_MARKED_UP_FRAMES_FOLDER_PATH_IMG = json_load[
    "output_marked_up_frames_folder_path_img"
]
OUTPUT_MARKED_UP_FRAMES_FOLDER_PATH_VID = json_load[
    "output_marked_up_frames_folder_path_vid"
]


# initialize event stream file
time_that_facdet_event_stream_began = datetime.datetime.utcnow()
time_that_individual_event_begins = ""
time_that_individual_event_ends = ""


# Getting Timestamp info
# https://www.geeksforgeeks.org/get-current-timestamp-using-python/
# beginning timestamp
stream_process = {
    "ProcessorType": "FaceDetector",
    "StreamStart": str(time_that_facdet_event_stream_began),
    "Events": [],
}

with open(EVENT_STREAM_PATH, "w") as outfile:
    json.dump(stream_process, outfile, indent=4)


def empty_image_folder():
    """Will take the path to the "output_marked_up_frames" in the project directory.
    This functon deletes all files in the output_marked_up_frames folder.
    Args:
        TEMP_FRAME_FOLDER_PATH (str): Path to "output_marked_up_frames" located in the directory.
    Return:
        -NONE-"""
    # https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
    path = OUTPUT_MARKED_UP_FRAMES_FOLDER_PATH_IMG
    for file_name in os.listdir(path):
        # construct full file path
        filepath = os.path.join(path, file_name)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)


def empty_vid_folder():
    """Will take the path to the "output_marked_up_frames" in the project directory.
    This functon deletes all files in the output_marked_up_frames folder.
    Args:
        TEMP_FRAME_FOLDER_PATH (str): Path to "output_marked_up_frames" located in the directory.
    Return:
        -NONE-"""
    # https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
    path = OUTPUT_MARKED_UP_FRAMES_FOLDER_PATH_VID
    for file_name in os.listdir(path):
        # construct full file path
        filepath = os.path.join(path, file_name)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)


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


def event_stream_writer(
    starting_time, ending_time, preview_list_dict, event_details_dict
) -> None:
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

    preview_list.append(preview_list_dict)
    event_details.append(event_details_dict)

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

    # facial_recognition
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", WEIGHTS_PATH, force_reload=True
    )

    # ----------for single image---------------------#

    # time when individual event begin(ending time is in the event steam writer function)

    time_that_individual_event_begins = datetime.datetime.utcnow()

    # Inference
    results = model(IMAGE_PATH)

    # Results
    results.show()
    # results.save()
    results.print()

    image_event_details_dict = {}
    # results.save(save_dir=OUTPUT_MARKED_UP_FRAMES_FOLDER_PATH + "/img")
    print(results.pandas().xyxy[0])  # im predictions (pandas)
    # 2 getting names and confidences from frame to later wirte to event stream
    for i in range(0, len(results.pandas().xyxy[0])):
        image_event_details_dict["name" + str(i)] = str(
            results.pandas().xyxy[0].iloc[i]["name"]
        )
        image_event_details_dict["conf" + str(i)] = str(
            results.pandas().xyxy[0].iloc[i]["confidence"]
        )
    if len(results.pandas().xyxy[0]) > 0:
        results.save(save_dir=OUTPUT_MARKED_UP_FRAMES_FOLDER_PATH_IMG + "/img")

        # output marked-up files information for event stream
        preview_dict = {
            "Type": "Image",
            "Path": OUTPUT_MARKED_UP_FRAMES_FOLDER_PATH_IMG + "/img",
        }

        time_that_individual_event_ends = datetime.datetime.utcnow()
        # Sending info to the event sream writer
        event_stream_writer(
            time_that_individual_event_begins,
            time_that_individual_event_ends,
            preview_dict,
            image_event_details_dict,
        )
    # -----------------------------------------------#

    # ------------for video frames--------------------#

    count = 0
    # for i in range(len(os.listdir(vid_path))):
    for i in range(407, 930):
        video_event_details_dict = {}

        # Images
        # img = path_to_front_of_head_pic  # or file, Path, PIL, OpenCV, numpy, list
        # img = pic_list[i]
        video_frame = VID_PATH + str(i) + ".jpg"

        time_that_individual_event_begins = datetime.datetime.utcnow()

        # Inference
        results = model(video_frame)

        # Results
        # results.show()
        # results.save()
        results.print()
        print(results.pandas().xyxy[0])  # im predictions (pandas)
        # print("dfl: "+str(len(results.pandas().xyxy[0])))  # im predictions (pandas)

        if len(results.pandas().xyxy[0]) > 0:
            # 1. save the frame to a file
            results.save(save_dir=OUTPUT_MARKED_UP_FRAMES_FOLDER_PATH_VID + "/img")
            print(results.pandas().xyxy[0].iloc[0]["name"])
            print(results.pandas().xyxy[0].iloc[0]["confidence"])
            # 2 getting names and confidences from frame to later wirte to event stream
            for i in range(0, len(results.pandas().xyxy[0])):
                video_event_details_dict["name" + str(i)] = str(
                    results.pandas().xyxy[0].iloc[i]["name"]
                )
                video_event_details_dict["conf" + str(i)] = str(
                    results.pandas().xyxy[0].iloc[i]["confidence"]
                )

            if count == 0:
                # output marked-up files information for event stream
                preview_dict = {
                    "Type": "Image",
                    "Path": OUTPUT_MARKED_UP_FRAMES_FOLDER_PATH_VID + "/img"
                }
            elif count > 0:
                # output marked-up files information for event stream
                preview_dict = {
                    "Type": "Image",
                    "Path": OUTPUT_MARKED_UP_FRAMES_FOLDER_PATH_VID + "/img" + str(count + 1)
                }

            time_that_individual_event_ends = datetime.datetime.utcnow()
            # Sending info to the event sream writer

            event_stream_writer(
                time_that_individual_event_begins,
                time_that_individual_event_ends,
                preview_dict,
                video_event_details_dict,
            )
            count += 1
    # -----------------------------------------------------------#


if __name__ == "__main__":
    delete_former_event_stream(EVENT_STREAM_PATH)
    empty_image_folder()
    empty_vid_folder()
    main()
