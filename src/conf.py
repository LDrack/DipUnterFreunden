# Created on 2021/03/24 - LD
# Configuration file for all KIVideo scripts.
# Import this to access all configuration variables,
# paths to models/weights/config files,

import os.path


##############################################################################
# Path to folders
##############################################################################
# This is your Project Root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELPATH = os.path.sep.join([ROOT_DIR, "model"])
INPUTPATH = os.path.sep.join([ROOT_DIR, "input"])
OUTPUTPATH = os.path.sep.join([ROOT_DIR, "output"])

# Paths to numes, cfg and weights files
labelsPath  = os.path.sep.join([MODELPATH, "indy.names"])
configPath  = os.path.sep.join([MODELPATH, "indy.cfg"])
weightsPath = os.path.sep.join([MODELPATH, "indy.weights"])


##############################################################################
# load the class labels our YOLO model was trained on
##############################################################################
LABELS = open(labelsPath).read().strip().split("\n")

##############################################################################
# The label indexes we want our AI to look for.
# For example, if we just want to look for a single class, use an array that
# contains only the number n
##############################################################################
objIdxAll = list(range(0, len(LABELS)))

##############################################################################
# Color constants for drawing with OpenCV (BGR)
##############################################################################
COLORS = [
(0, 255, 0),        # Normal        Green
(0, 0, 255),        # NoHat         Red
(255, 0, 0),        # NoFace        Blue
(255, 255, 0),      # NoLeg         Cyan
(255, 0, 255),      # NoBodyPrint   Magenta
(0, 255, 255),      # NoHand        Yellow
(255, 165, 0),      # NoHead        Orange
(153, 50, 204)      # NoArm         Purple
]

# colorGreen    = (0, 255, 0),        # Normal        Green
# colorRed      = (0, 0, 255),        # NoHat         Red
# colorBlue     = (255, 0, 0),        # NoFace        Blue
# colorCyan     = (255, 255, 0),      # NoLeg         Cyan
# colorMagenta  = (255, 0, 255),      # NoBodyPrint   Magenta
# colorYellow   = (0, 255, 255),      # NoHand        Yellow
# colorOrange   = (255, 165, 0),      # NoHead        Orange
# colorPurple   = (153, 50, 204)      # NoArm         Purple


##############################################################################
# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
##############################################################################
MIN_CONF = 0.3
NMS_THRESH = 0.3


##############################################################################
# boolean indicating if NVIDIA CUDA GPU should be used
##############################################################################
USE_GPU = False




