#TODO: no original categories
# shortened version only

import copy
import numpy as np


# After merging, our label-id to class (string);
class_names = [
    "cabinet", "refrigerator", "shelf", "stove", "bed", # 0..5
    "sink", "washer", "toilet", "bathtub", "oven", # 5..10
    "dishwasher", "fireplace", "stool", "chair", "table", # 10..15
    "tv_monitor", "sofa", # 15..17
]

# 3D Anchor-sizes of merged categories (dx, dy, dz)
'''
   Anchor box sizes are computed based on box corner order below:
       6 -------- 7
      /|         /|
     5 -------- 4 .
     | |        | |
     . 2 -------- 3
     |/         |/
     1 -------- 0 
'''


class ARKitDatasetConfig(object):
    def __init__(self):
        """
        init will set values for:
            self.class_names
            self.cls2label (after mapping)
            self.label2cls (after mapping)
            self.num_class

        Args:
        """
        # final training/val categories
        self.class_names = class_names
        self.label2cls = {}
        self.cls2label = {}
        for i, cls_ in enumerate(class_names):
            self.label2cls[i] = cls_
            self.cls2label[cls_] = i

        self.num_class = len(self.class_names)
