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

MODELPATH = os.path.sep.join([ROOT_DIR, "../model"])
INPUTPATH = os.path.sep.join([ROOT_DIR, "input"])
OUTPUTPATH = os.path.sep.join([ROOT_DIR, "output"])

# Paths to numes, cfg and weights files
labelsPath  = os.path.sep.join([MODELPATH, "indy1.names"])
configPath  = os.path.sep.join([MODELPATH, "indy1.cfg"])
weightsPath = os.path.sep.join([MODELPATH, "indy2.weights"])


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
(153, 50, 204),     # NoArm         Purple
(255, 0, 255),      # NoBodyPrint   Magenta
(255, 0, 0),        # NoFace        Blue
(0, 255, 255),      # NoHand        Yellow
(0, 0, 255),        # NoHat         Red
(255, 255, 255),    # NoHead        White
(255, 255, 0),      # NoLeg         Cyan
(0, 255, 0),        # Normal        Green
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
# boolean indicating if NVIDIA CUDA GPU should be used
##############################################################################
USE_GPU = False




