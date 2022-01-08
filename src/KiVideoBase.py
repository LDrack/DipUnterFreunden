# Base class for all KIVideo implementations.
# Definies common init() and processFrame() functions.
# Overwrite processFrame() to change AI algorithm, include trackers etc.
# Classes that inherit from this base class are meant to be used in the GUI.
# For a runnable version of this script, see KIVideoStandalone.py

from object_detection import detect_objects
import conf
import cv2.cv2 as cv2
from collections import deque
# import imutils

class KiVideoBase:
	args = None
	net = None
	ln = None
	vs = None
	writer = None
	tracker = None
	initBB = None
	ballvector = None
 
	def init(self):
		##################################################################################
		# Get an array of all arguments the script was started with
		##################################################################################
		self.args = conf.getArguments()

		##################################################################################
		# Load our YOLO object detector trained on custom dataset.
		# This uses OpenCV as backend and uses the Darknet framework.
		# Change the paths below to chose the desired AI - ball and player,
		# ball only, ...
		# If CUDA was enabled in conf.py, the AI will run on your GPU.
		##################################################################################
		print("[INFO] loading YOLO from disk...")
		self.net = cv2.dnn.readNetFromDarknet(conf.configPath, conf.weightsPath)
		if conf.USE_GPU:
			print("[INFO] setting preferable backend and target to CUDA...")
			self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
			self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
		
		##################################################################################
		# Determine only the *output* layer names that we need from YOLO
		##################################################################################
		self.ln = self.net.getLayerNames()
		self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

		##################################################################################
		# Initialize video stream, pointer to output video file, object tracker,
		# bounding box coordinates of the tracked object
		##################################################################################
		self.vs = cv2.VideoCapture(self.args["input"] if self.args["input"] else 0)
  
  		##################################################################################
		# Initialize ballvector
		##################################################################################
		self.ballvectorLength 	= 64
		self.ballvector = deque(maxlen=self.ballvectorLength)
     

	def processFrame(self, frame):
		# resize the frame and then detect given objects in it
		# frame = imutils.resize(frame, width=720)
		results = detect_objects(frame, self.net, self.ln, conf.objIdxBallPlayer)

		# loop over the results
		for (i, (prob, bbox, centroid, classId)) in enumerate(results):
			# extract the bounding box and centroid coordinates, then
			# initialize the color of the annotation
			(startX, startY, endX, endY) = bbox
	
			# Set color depending on class: ball is red, players are green
			if classId == conf.labelsBallPlayer.index("player"):
				color = (0, 0, 255)
			else:
				color = (0, 255, 0)
			# draw a bounding box around the object
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


