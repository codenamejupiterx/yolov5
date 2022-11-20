#SOURCE: https://github.com/ultralytics/yolov5
#helpful link: https://github.com/sondrion/SMOKE_CLASS_DETECTION
#train custom data link:  https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
#training command (make sure you use python3 not just python and add --workers 0 to the end example below:
# #python3 train.py --img 640 --batch 16 --epochs 300 --data /Users/benjaminhall/Documents/GitHub/yolov5/data/head_detect_dataset/data.yaml --weights yolov5s.pt --workers 0)
#-------------------------------------------------------------------------------------------------------------------------

import torch

# Model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
model = torch.hub.load('ultralytics/yolov5', 'custom','/Users/benjaminhall/Documents/GitHub/yolov5/runs/train/exp9/weights/best.pt', force_reload=True)

path_to_back_of_head_pic = "/Users/benjaminhall/Desktop/crowd_heads_back.png"
path_to_front_of_head_pic = "/Users/benjaminhall/Desktop/crowd_heads_front.png"
path_to_back_of_head_pic_1 = "/Users/benjaminhall/Downloads/back_of_heads_1.jpg"
path_to_side_of_head_pic_1 = "/Users/benjaminhall/Downloads/sides_of_heads_1.jpg"
path_to_front_and_side_of_head_pic_1 = "/Users/benjaminhall/Downloads/side_and_front_of_head_pic_1.jpg"

pic_list = [path_to_back_of_head_pic_1,path_to_side_of_head_pic_1,path_to_front_and_side_of_head_pic_1]

for i in range(len(pic_list)):

    # Images
    #img = path_to_front_of_head_pic  # or file, Path, PIL, OpenCV, numpy, list
    img = pic_list[i]

    # Inference
    results = model(img)

    # Results
    results.show()
    results.print()
    """ print(results)

    import wandb
    wandb.login() """
