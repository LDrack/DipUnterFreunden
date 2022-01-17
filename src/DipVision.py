# DIP unter Freunden.


from object_detection import detect_objects
import conf
import cv2.cv2 as cv2
import os
# import imutils

class DipVision:
	args = None
	net = None
	ln = None
	vs = None
	writer = None
	tracker = None
	initBB = None
 
	def init(self):
		##################################################################################
		# Load our YOLO object detector trained on custom dataset.
		# This uses OpenCV as backend and uses the Darknet framework.
		# Change the paths below to chose the desired AI model.
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
		self.ln = [self.ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

     

	########################################################################
	# Processes a single frame by
	#   - applying object detection on it
	#   - drawing any found objects into the frame in-place
	########################################################################
	def processFrame(self, frame):
		# resize the frame and then detect given objects in it
		# frame = imutils.resize(frame, width=720)
		results = detect_objects(frame, self.net, self.ln, conf.objIdxAll)

		# loop over the results
		for (i, (prob, bbox, centroid, classId)) in enumerate(results):
			# extract the bounding box and centroid coordinates, then
			# initialize the color of the annotation
			(startX, startY, endX, endY) = bbox
	
			# Set color depending on class: Normal Indy is green, faulty is red, orange, blue, etc.
			color = conf.COLORS[classId]
			# draw a bounding box around the object
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			# write the class name and certainty above the box
			text = "{}: {:.4f}".format(conf.LABELS[classId], prob)
			cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


	########################################################################
	# Handles user input to quit, skip forwards/backwards with z and v keys
	# Returns true if user pressed Q, else false
	########################################################################
	def handleKeypress(self):
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			# Break from the loop
			return True
		# elif key == ord("z"):
		# 	# Z für Zukunft
		# 	framenr = self.vs.get(cv2.CAP_PROP_POS_FRAMES)
		# 	framenr = framenr + 1000
		# 	self.vs.set(cv2.CAP_PROP_POS_FRAMES, framenr)
		# elif key == ord("v"):
		# 	# V für Vergangenheit
		# 	framenr = self.vs.get(cv2.CAP_PROP_POS_FRAMES)
		# 	framenr = (framenr - 1000) if framenr >= 1000 else 0
		# 	self.vs.set(cv2.CAP_PROP_POS_FRAMES, framenr)
		return False



	##################################################################################
	# Main loop: go over the frames in the image folder
	##################################################################################
	def run(self):
		# iterate through the names of contents of the folder
		for img_filename in os.listdir(conf.INPUTPATH):

			# create the full input path and read the image file
			img_path = os.path.join(conf.INPUTPATH, img_filename)
			img = cv2.imread(img_path)
			
			self.processFrame(img)
		
			# show the output frame
			cv2.imshow("DipVision", img)
			if self.handleKeypress():
				# User pressed q, so break from the loop
				break
   
			# TODO: save resulting image and data (number of defects, ect.)

		cv2.destroyAllWindows()






if __name__ == "__main__":
	inst = DipVision()
	inst.init()
	inst.run()
