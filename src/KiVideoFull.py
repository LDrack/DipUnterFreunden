# Final class that gets included in the GUI.

from KiVideo.KIVideoBase import KIVideoBase
from KiVideo.KIVideoRunnableMixin import KIVideoRunnableMixin
from KiVideo.object_detection import detect_objects
from KiVideo import conf
import cv2.cv2 as cv2
from collections import deque
import numpy as np
from scipy.spatial import distance
# import imutils

class KIVideoFull(KIVideoBase, KIVideoRunnableMixin):
	title = "KIVideoFull"
    
	def init(self):
		super().init()
		self.drawPlayers 		= True
		self.drawBall 			= True
		self.drawBallvector 	= True
		self.drawClosestPlayer 	= False
  

	##############################################################################
	# Turn on/off GUI elements
	##############################################################################
	def drawPlayersOn(self):
		self.drawPlayers = True
	def drawPlayersOff(self):
		self.drawPlayers = False

	def drawBallOn(self):
		self.drawBall = True
	def drawBallOff(self):
		self.drawBall = False

	def drawBallvectorOn(self):
		self.drawBallvector = True
		self.ballvector.clear()
	def drawBallvectorOff(self):
		self.drawBallvector = False
		self.ballvector.clear()
  
	def drawClosestPlayerOn(self):
		self.drawClosestPlayer = True
	def drawClosestPlayerOff(self):
		self.drawClosestPlayer = False

	def changeBallvectorLength(self, len):
		self.ballvectorLength = len
		self.ballvector = deque(maxlen=self.ballvectorLength)


	##############################################################################
	# Main function that does the actual work: run AI, draw rectangles, ...
	##############################################################################
	def processFrame(self, frame):
		# resize the frame and then detect given objects in it
		# frame = imutils.resize(frame, width=720)
		results = detect_objects(frame, self.net, self.ln, conf.objIdxBallPlayer)

		if self.drawClosestPlayer:
			closestPlayerIdx = self._calcClosestPlayer(results)
		else:
			closestPlayerIdx = set()

		# loop over the results
		for (i, (prob, bbox, centroid, classId)) in enumerate(results):
			# extract the bounding box and centroid coordinates, then
			# initialize the color of the annotation
			(startX, startY, endX, endY) = bbox

			# Set color depending on class: ball is red, players are green
			# If option to mark closest player is enabled: ball and closest player are marked blue
			if self.drawPlayers and classId == conf.labelsBallPlayer.index("player"):
				color = (255, 0, 0) if i in closestPlayerIdx else (0,0,255)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
				# Add ball coordinates to ballvector
				if self.drawBallvector:
					self._addToBallvector(centroid)
					self._drawBallvector(frame)
			elif self.drawBall and classId == conf.labelsBallPlayer.index("ball"):
				color = (255, 0, 0) if i in closestPlayerIdx else (0,255,0)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			
			
    
    ##############################################################################
	# Update the ballvector with the centerpoint of tracked object
	##############################################################################
	def _addToBallvector(self, center):
		self.ballvector.appendleft(center)

	##############################################################################
	# loop over the set of tracked points
	##############################################################################
	def _drawBallvector(self, frame):
		for i in range(1, len(self.ballvector)):
			# if either of the tracked points are None, ignore them
			if self.ballvector[i - 1] is None or self.ballvector[i] is None:
				continue
			# otherwise, compute the thickness of the line and draw the connecting lines
			thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
			color = (0, 0, 255 - (2*i))
			cv2.line(frame, self.ballvector[i - 1], self.ballvector[i], color, thickness)

	##############################################################################
	# Returns a set that contains the indexes of the ball and the closest player.
	##############################################################################
	def _calcClosestPlayer(self, results):
		idxs = set()
		# extract centroids of (1) all objects (players+ball) and (2) the ball from results
		all_centroids = np.array([r[2] for r in results])
		ball_centroids = np.array([r[2] for r in results if r[3] == 0])

		# Compute the euclidian distances between the ball and each player
		if len(ball_centroids) == 1:
			D = distance.cdist(all_centroids, ball_centroids, metric="euclidean")
			ballIdx = np.where(D == np.amin(D))[0][0]		# Index of ball
			D[ballIdx] = np.amax(D)							# Overwrite Index of ball
			minDistIdx = np.where(D == np.amin(D))[0][0]	# Calc minimum distance of other centroids
			idxs.add(ballIdx)								# Add ball index
			idxs.add(minDistIdx)							# Add closest player index
		return idxs


if __name__ == "__main__":
	inst = KIVideoFull()
	inst.init()
	inst.run()
