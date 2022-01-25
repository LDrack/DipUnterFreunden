# DIP unter Freunden.

from object_detection import detect_objects
import conf
import cv2
import os


class DipVision:
	args = None
	net = None
	ln = None
	vs = None
	writer = None
	tracker = None
	initBB = None
 
	def init(self):
		########################################################################
		# Load our YOLO object detector trained on custom dataset.
		# This uses OpenCV as backend and uses the Darknet framework.
		# Change the paths below to chose the desired AI model.
		# If CUDA was enabled in conf.py, the AI will run on your GPU.
		########################################################################
		print("[INFO] loading YOLO from disk...")
		self.net = cv2.dnn.readNetFromDarknet(conf.configPath, conf.weightsPath)
		if conf.USE_GPU:
			print("[INFO] setting preferable backend and target to CUDA...")
			self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
			self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
		
		########################################################################
		# Determine only the *output* layer names that we need from YOLO
		########################################################################
		self.ln = self.net.getLayerNames()
		self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

		########################################################################
		# Flags determining how images are shown to user
		########################################################################
		self.automaticMode = False	# Images are shown like a slideshow (fast)
		self.endMode = False		# Images are not shown, but results are
									# processed until the end

     

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
	# A turns on automatic mode, which works like a slideshow.
	# E turns on end mode, which skips visualization but processes all images
	########################################################################
	def handleKeypress(self, i):
		# Automatic mode does not wait for user keypress
		if (self.automaticMode):
			key = cv2.waitKey(1) & 0xFF
			newI = i+1
		else:
			key = cv2.waitKey(0) & 0xFF
			newI = i

		if key == ord("q"):
			# Q für Quittieren
			newI = self.nOfImages
		elif key == ord("z"):
			# Z für Zukunft
			newI = i+1
			self.automaticMode = False
		elif key == ord("v"):
			# V für Vergangenheit
			newI = max(0, i-1)
			self.automaticMode = False
		elif key == ord("a"):
			# A für Automatik
			newI = i+1
			self.automaticMode = True
		elif key == ord("e"):
			# E für Ende
			newI = i+1
			self.endMode = True
			cv2.destroyAllWindows()
		return newI



	##################################################################################
	# Main loop: go over the frames in the image folder
	##################################################################################
	def run(self):
		# iterate through the names of contents of the folder
		img_filenames = os.listdir(conf.INPUTPATH)
		self.nOfImages = len(img_filenames)
		i = 0

		while i < self.nOfImages:
			# create the full input path and read the image file
			img_path = os.path.join(conf.INPUTPATH, img_filenames[i])
			img = cv2.imread(img_path)
			
			self.processFrame(img)

			# TODO: save resulting image and data (number of defects, ect.)
			# >>>>>>
			img_output_path = os.path.join(conf.OUTPUTPATH, img_filenames[i])
			cv2.imwrite(img_output_path, img)


			# If user pressed e, skip visualization and process all images immediately
			# Otherwise, show the image and handle user input
			if (self.endMode):
				i = i+1
			else:
				cv2.imshow("DipVision", img)
				i = self.handleKeypress(i)


		cv2.destroyAllWindows()






if __name__ == "__main__":
	inst = DipVision()
	inst.init()
	inst.run()
