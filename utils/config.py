import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEY")
API_BASE=os.getenv("API_BASE")
API_NAME = "TrafficLens"

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH")
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH")
DATA_ROOT = os.getenv("DATA_ROOT")
MODEL_PATH = os.getenv("MODEL_PATH")
epochs = 15

class_mappings = {
    0: "20 speed limit",
    1: "30 speed limit",
    2: "50 speed limit",
    3: "60 speed limit",
    4: "70 speed limit",
    5: "80 speed limit",
    6: "End 80 speed limit",
    7: "100 speed limit",
    8: "120 speed limit",
    9: "No passing",
    10: "No passing for vehicles over 3.5t",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "No vehicles over 3.5t",
    17: "No entry",
    18: "General warning",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Ice/snow warning",
    31: "Wild animals",
    32: "End speed + passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout",
    41: "End no passing",
    42: "End no passing vehicle > 3.5t",
}
