# Runnable version of KIVideoBase. Implements run() as main loop
# and handles user input and video output.

# import KIVideoBase
# from object_detection import detect_objects
# import conf
# import numpy as np
# import imutils
import cv2.cv2 as cv2

class KIVideoRunnableMixin:
	title = "Frame"

	##################################################################################
	# Main loop: go over the frames from the video stream
	##################################################################################
	def run(self):
		while True:
			(grabbed, frame) = self.vs.read()	# read the next frame from the file
			if not grabbed:
				break						# end of the stream reached
			
			self.processFrame(frame)
		
			# check to see if the output frame should be displayed to our screen
			if self.args["display"] > 0:
				# show the output frame
				cv2.imshow(self.title, frame)
				if self._handleKeypress():
					# User pressed q, so break from the loop
					break
   
			self._writeOutputVideo(frame)

		self.vs.release()
		cv2.destroyAllWindows()

	########################################################################
	# Handles user input to quit, skip forwards/backwards
	# Returns true if user pressed Q, else false
	########################################################################
	def _handleKeypress(self):
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			# Break from the loop
			return True
		elif key == ord("z"):
			# Z für Zukunft
			framenr = self.vs.get(cv2.CAP_PROP_POS_FRAMES)
			framenr = framenr + 1000
			self.vs.set(cv2.CAP_PROP_POS_FRAMES, framenr)
			self.ballvector.clear()
		elif key == ord("v"):
			# V für Vergangenheit
			framenr = self.vs.get(cv2.CAP_PROP_POS_FRAMES)
			framenr = (framenr - 1000) if framenr >= 1000 else 0
			self.vs.set(cv2.CAP_PROP_POS_FRAMES, framenr)
			self.ballvector.clear()
		return False

	########################################################################
	# Writes a single frame to the specified output file path
	########################################################################
	def _writeOutputVideo(self, frame):
		if self.args["output"] != "":
			# if an output video file path has been supplied and the video
			# writer has not been initialized, do so now
			if self.writer is None:
				# initialize our video writer
				fourcc = cv2.VideoWriter_fourcc(*"mp4v")
				self.writer = cv2.VideoWriter(self.args["output"], fourcc, 25,
					(frame.shape[1], frame.shape[0]), True)
			else:
				self.writer.write(frame)


########################################################################
# Copy these to make the script runnable:
########################################################################
# inst = KIVideoClassname()
# inst.init()
# inst.run()
