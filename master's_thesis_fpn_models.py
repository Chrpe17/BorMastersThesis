# -*- coding: utf-8 -*-
"""Master's thesis - FPN models.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kneXikXcCeI5-nZxNIfxbfRYzZgfsVE5

# Install Detectron2 and its dependencies
"""

!pip install pyyaml==5.1
!pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html

"""# Imports"""

# Commented out IPython magic to ensure Python compatibility.
import torch
assert torch.__version__.startswith("1.8") 
import torchvision
import cv2
from google.colab import drive

#Import some common imports
import itertools
import os
import numpy as np
import json
import random
import copy
import matplotlib.pyplot as plt
# %matplotlib inline

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

"""# Prepare the dataset

## Mount to drive
"""

drive.mount('/content/drive') #This step should be avoided if data is not placed on drive

"""Note that the following code has taken inspiration from different resources: 
The function to convert data to coco format has taken inspiration from the "Detectron 2 compare models + augmentation" kaggle. URL:https://www.kaggle.com/code/dhiiyaur/detectron-2-compare-models-augmentation

## Make function to convert data to COCO format
"""

def get_data_dicts(directory, classes):
    dataset_dicts = []
    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}
        
        filename = os.path.join(directory, img_anns["imagePath"])
        
        record["file_name"] = filename
        record["image_id"] = filename
        record["height"] = 480
        record["width"] = 640
      
        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']] # x coord
            py = [a[1] for a in anno['points']] # y-coord
            poly = [(x, y) for x, y in zip(px, py)] # poly 
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(anno['label']),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

"""## Load data"""

#List the five classes 
classes = ['sink', 'door', 'bed', 'screen', 'socket']

#Give path to data - if path is not on drive change the path before running
data_path = '/content/drive/MyDrive/robotdata/Blue_ocean_dataset1/'

for d in ["train", "valid", "test"]:
    DatasetCatalog.register(
        "robot_" + d, 
        lambda d=d: get_data_dicts(data_path+d, classes) #uses get_data_dicts function to make data into COCO format
    )
    MetadataCatalog.get("robot_" + d).set(thing_classes=classes)

#Collect the data into metadatacatalog 
microcontroller_metadata = MetadataCatalog.get("robot_train")

#Convert trainingset to COCO format and save as train_dicts
train_dicts = get_data_dicts(data_path+'train', classes)

#Convert validationset to COCO format and save as valid_dicts
valid_dicts = get_data_dicts(data_path+'valid', classes)

"""### Visualize loaded data"""

#Visualize for checking the data is loaded properly 
for d in random.sample(train_dicts, 5):
    img = cv2.imread(d["file_name"])
    v = Visualizer(img[:, :, ::-1], metadata=microcontroller_metadata, scale=0.5)
    v = v.draw_dataset_dict(d)
    plt.figure(figsize = (14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.show()

"""# Testing 4 chosen FPN models

## R101 - FPN

### Train on baseline model
"""

cfg_fpn_r101 = get_cfg()
cfg_fpn_r101.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")) #Load model
cfg_fpn_r101.DATASETS.TRAIN = ('robot_train',)
cfg_fpn_r101.DATASETS.TEST = ()  
cfg_fpn_r101.DATALOADER.NUM_WORKERS = 2
cfg_fpn_r101.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml") #Load weights
cfg_fpn_r101.SOLVER.IMS_PER_BATCH = 4 # Set to 4 to run on collab due to limited resources
cfg_fpn_r101.SOLVER.MAX_ITER = 1000
cfg_fpn_r101.SOLVER.STEPS = []
cfg_fpn_r101.MODEL.ROI_HEADS.NUM_CLASSES = 5 #Number of classes

os.makedirs(cfg_fpn_r101.OUTPUT_DIR, exist_ok=True) #Make output directory 
trainer_fpn_r101 = DefaultTrainer(cfg_fpn_r101) #Use default trainer
trainer_fpn_r101.resume_or_load(resume=False)
trainer_fpn_r101.train()# train model

"""#### Tensorboard over training"""

# Commented out IPython magic to ensure Python compatibility.
# Look at training curves in tensorboard:
# %load_ext tensorboard
# %tensorboard --logdir output

"""### Test and evaluation"""

cfg_fpn_r101.MODEL.WEIGHTS = os.path.join(cfg_fpn_r101.OUTPUT_DIR, "model_final.pth")
cfg_fpn_r101.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 #Set threshold
cfg_fpn_r101.DATASETS.TEST = ("robot_valid",) #Give validation set to predict on
predictor_fpn_r101 = DefaultPredictor(cfg_fpn_r101) #Make predictions

"""#### Evaluation matrix"""

evaluator_fpn_r101 = COCOEvaluator("robot_valid", output_dir="./output") #Use COCO evaluator
val_loader_fpn_r101 = build_detection_test_loader(cfg_fpn_r101, "robot_valid") #Give validation set
print(inference_on_dataset(predictor_fpn_r101.model, val_loader_fpn_r101, evaluator_fpn_r101)) #Print predictions

"""#### Visualization of predictions"""

#Gives five random predictions from prediction
for d in random.sample(valid_dicts, 5):       
    im = cv2.imread(d["file_name"])
    outputs = predictor_fpn_r101(im) 
    v = Visualizer(im[:, :, ::-1],
                   metadata=microcontroller_metadata, 
                   scale=0.5
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(14, 10))
  plt.imshow(out.get_image()[:, :, ::-1][..., ::-1])

"""### Save model"""

torch.save(cfg_fpn_r101, 'model_final.pth')

"""## X101-FPN

### Train baseline model
"""

cfg_fpn_x101 = get_cfg()
cfg_fpn_x101.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")) #Load model
cfg_fpn_x101.DATASETS.TRAIN = ('robot_train',)
cfg_fpn_x101.DATASETS.TEST = ()  
cfg_fpn_x101.DATALOADER.NUM_WORKERS = 2
cfg_fpn_x101.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml") #Load weights
cfg_fpn_x101.SOLVER.IMS_PER_BATCH = 4 # Set to four to make collab able to run due to limited resources
cfg_fpn_x101.SOLVER.MAX_ITER = 1000
cfg_fpn_x101.SOLVER.STEPS = []
cfg_fpn_x101.MODEL.ROI_HEADS.NUM_CLASSES = 5 #number of classes

os.makedirs(cfg_fpn_x101.OUTPUT_DIR, exist_ok=True) #Make directory for output
trainer_fpn_x101 = DefaultTrainer(cfg_fpn_x101) #Use default trainer
trainer_fpn_x101.resume_or_load(resume=False)
trainer_fpn_x101.train() #Train model

"""#### Tensorboard over training"""

# Commented out IPython magic to ensure Python compatibility.
# Look at training curves in tensorboard:
# %load_ext tensorboard
# %tensorboard --logdir output

"""### Test and evaluation"""

cfg_fpn_x101.MODEL.WEIGHTS = os.path.join(cfg_fpn_x101.OUTPUT_DIR, "model_final.pth")
cfg_fpn_x101.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # Set threshold
cfg_fpn_x101.DATASETS.TEST = ("robot_valid",) #give validation set
predictor_fpn_x101 = DefaultPredictor(cfg_fpn_x101) # Predict on validation set

"""#### Evaluation matrix"""

evaluator_fpn_x101 = COCOEvaluator("robot_valid", output_dir="./output") #Use coco evaluator
val_loader_fpn_x101 = build_detection_test_loader(cfg_fpn_x101, "robot_valid") #Predict on validationset
print(inference_on_dataset(predictor_fpn_x101.model, val_loader_fpn_x101, evaluator_fpn_x101)) #Print evaluation matrix

"""#### Visualization of predictions"""

#Gives five random predictions from prediction
for d in random.sample(valid_dicts, 5):       
    im = cv2.imread(d["file_name"])
    outputs = predictor_fpn_r101(im) 
    v = Visualizer(im[:, :, ::-1],
                   metadata=microcontroller_metadata, 
                   scale=0.5
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(14, 10))
  plt.imshow(out.get_image()[:, :, ::-1][..., ::-1])

"""### Save model"""

torch.save(cfg_fpn_x101, 'model_final.pth')

"""## R101-Retina

### Train on baseline model
"""

cfg_fpn_re = get_cfg()
cfg_fpn_re.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml")) #Load model
cfg_fpn_re.DATASETS.TRAIN = ('robot_train',)
cfg_fpn_re.DATASETS.TEST = ()  
cfg_fpn_re.DATALOADER.NUM_WORKERS = 2
cfg_fpn_re.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml") #Load weights
cfg_fpn_re.SOLVER.IMS_PER_BATCH = 4 #set to four to be able to run on collab due to limited resources
cfg_fpn_re.SOLVER.MAX_ITER = 1000
cfg_fpn_re.SOLVER.STEPS = []
cfg_fpn_re.MODEL.RETINANET.NUM_CLASSES = 5 # Number of classes

os.makedirs(cfg_fpn_re.OUTPUT_DIR, exist_ok=True) #create directory for output
trainer_fpn_re = DefaultTrainer(cfg_fpn_re) # Use default trainer
trainer_fpn_re.resume_or_load(resume=False)
trainer_fpn_re.train() # Train model

"""#### Tensorboard over training"""

# Commented out IPython magic to ensure Python compatibility.
# Look at training curves in tensorboard:
# %load_ext tensorboard
# %tensorboard --logdir output

"""### Test and evaluation"""

cfg_fpn_re.MODEL.WEIGHTS = os.path.join(cfg_fpn_re.OUTPUT_DIR, "model_final.pth")
cfg_fpn_re.MODEL.RETINANET.SCORE_THRESH_TEST = 0.7 #Set threshold
cfg_fpn_re.DATASETS.TEST = ("robot_valid",) #Give validation set
predictor_fpn_re = DefaultPredictor(cfg_fpn_re) # Predict on validation set

"""#### Evaluation matrix"""

evaluator_fpn_re = COCOEvaluator("robot_valid", output_dir="./output") #Use COCO evaluator
val_loader_fpn_re = build_detection_test_loader(cfg_fpn_re, "robot_valid") # Evaluate on validation set
print(inference_on_dataset(predictor_fpn_re.model, val_loader_fpn_re, evaluator_fpn_re)) #print evaluation

"""#### Visualization of predictions"""

#Gives five random predictions from prediction
for d in random.sample(valid_dicts, 5):       
    im = cv2.imread(d["file_name"])
    outputs = predictor_fpn_r101(im) 
    v = Visualizer(im[:, :, ::-1],
                   metadata=microcontroller_metadata, 
                   scale=0.5
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(14, 10))
  plt.imshow(out.get_image()[:, :, ::-1][..., ::-1])

"""### Save model"""

torch.save(cfg_fpn_re, 'model_final.pth')

"""## R50-FPN

### Train on baseline model
"""

cfg_fpn_fast_r50 = get_cfg()
cfg_fpn_fast_r50.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")) # Load model
cfg_fpn_fast_r50.DATASETS.TRAIN = ('robot_train',)
cfg_fpn_fast_r50.DATASETS.TEST = ()  
cfg_fpn_fast_r50.DATALOADER.NUM_WORKERS = 2
cfg_fpn_fast_r50.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml") # Load weights
cfg_fpn_fast_r50.SOLVER.IMS_PER_BATCH = 4 #set to four to make collab able to run due to limited resources
cfg_fpn_fast_r50.SOLVER.MAX_ITER = 1000
cfg_fpn_fast_r50.SOLVER.STEPS = []
cfg_fpn_fast_r50.MODEL.ROI_HEADS.NUM_CLASSES = 5 #number of classes

os.makedirs(cfg_fpn_fast_r50.OUTPUT_DIR, exist_ok=True) #Make directory for output from model
trainer_fpn_fast_r50 = DefaultTrainer(cfg_fpn_fast_r50) # Use default trainer
trainer_fpn_fast_r50.resume_or_load(resume=False)
trainer_fpn_fast_r50.train() #Train model

"""#### Tensorboard over training"""

# Commented out IPython magic to ensure Python compatibility.
# Look at training curves in tensorboard:
# %load_ext tensorboard
# %tensorboard --logdir output

"""### Test and evaluation"""

cfg_fpn_fast_r50.MODEL.WEIGHTS = os.path.join(cfg_fpn_fast_r50.OUTPUT_DIR, "model_final.pth")
cfg_fpn_fast_r50.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 #Set threshold
cfg_fpn_fast_r50.DATASETS.TEST = ("robot_valid",) #Give validationset
predictor_fpn_fast_r50 = DefaultPredictor(cfg_fpn_fast_r50) #Predict on validationset

"""#### Evaluation matrix"""

evaluator_fpn_fast_r50 = COCOEvaluator("robot_valid", output_dir="./output") #Use COCO evaluator
val_loader_fpn_fast_r50 = build_detection_test_loader(cfg_fpn_fast_r50, "robot_valid") #Evaluate on validationset
print(inference_on_dataset(predictor_fpn_fast_r50.model, val_loader_fpn_fast_r50, evaluator_fpn_fast_r50)) #Print evaluation

"""#### Visualization of predictions"""

#Gives five random predictions from prediction
for d in random.sample(valid_dicts, 5):       
    im = cv2.imread(d["file_name"])
    outputs = predictor_fpn_r101(im) 
    v = Visualizer(im[:, :, ::-1],
                   metadata=microcontroller_metadata, 
                   scale=0.5
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(14, 10))
  plt.imshow(out.get_image()[:, :, ::-1][..., ::-1])

"""### Save model"""

torch.save(cfg_fpn_fast_r50, 'model_final.pth')