import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import glob
import cv2
from sklearn.model_selection import train_test_split

human_angry = glob.glob("../Data/Human/Angry/*")