# Object Detection Using Deep Learning...

import kagglehub
alincijov_self_driving_cars_path = kagglehub.dataset_download('alincijov/self-driving-cars')
print('Data source import complete.')

import numpy as np
import pandas as pd
import cv2
from sklearn.utils import shuffle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')

df = pd.read_csv(f'{alincijov_self_driving_cars_path}/labels_train.csv')
df = shuffle(df)
df.head()

classes = df.class_id.unique()
print(classes)

boxes = {}
images = {}
base_path = f'{alincijov_self_driving_cars_path}/images/'
for class_id in classes:
  first_row = df[df['class_id'] == class_id].iloc[0]
  images[class_id] = cv2.imread(base_path + first_row['frame'])
  boxes[class_id] = [first_row['xmin'],first_row['xmax'],first_row['ymin'],first_row['ymax']]

labels = {
    1: 'Label 1',
    2: 'Lable 2',
    3: 'Lable 3',
    4: 'Lable 4',
    5: 'Lable 5',
}
for i in classes:
  xmin, xmax, ymin, ymax = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
  plt.figure(figsize=(8, 10))
  plt.title("Lable " + labels[i])
  plt.imshow(images[i])
  plt.gca().add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='yellow', fill=False, linewidth=2))
  plt.show()

!pip install ultralytics
from ultralytics import YOLO
import PIL
from PIL import Image
from IPython.display import display
import os
import pathlib

model = YOLO("yolov8m.pt")

image_path = f'{alincijov_self_driving_cars_path}/images/{df.iloc[0]["frame"]}'
results=model.predict(source=image_path, save=True, conf=0.2, iou=0.5)

result = results[0]
box = result.boxes[0]

for result in results:
  boxes = result.boxes
  masks = result.masks
  probs = result.probs

cords = box.xyxy[0].tolist()
class_id = box.cls[0].item()
conf = box.conf[0].item()
print("Object type:", class_id)
print("Coordinates:", cords)
print("Probability:", conf)

for box in result.boxes:
  class_id = result.names[box.cls[0].item()]
  cords = box.xyxy[0].tolist()
  cords = [round(x) for x in cords]
  conf = round(box.conf[0].item(),2)
  print("Object type:", class_id)
  print("Coordinates:", cords)
  print("Probability:", conf)
  print("---")

results1 = model.predict(source=f'{alincijov_self_driving_cars_path}/images/1478020211690815798.jpg',
              save=True, conf=0.2, iou=0.5)
Results = results1[0]

plot = results1[0].plot()
plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
display(Image.fromarray(plot))
