# importing the needed bindings
import numpy as np
import pywt
import cv2

class DescribeTexture:

	def describe_texture(self, img):
		# converting the given image to grayscale and normalizing it
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		imArray =  np.float32(gray_img)   
    		imArray /= 255;

		# initializing arrays to store mean, variance and final features
		imMean, imVar, feats = [], [], []
		
		# calculating the dimensions of the regions, for which mean and variance are to be calculated
		# we split the image into 8X8 separate regions
		(h, w) = gray_img.shape[:2]
		(cX, cY) = (int(w * 0.125), int(h * 0.125))

		# iterating over all of the 64 separate regions
		for r in range(8):
		    imRMean, imRVar = [],[]
        	    for c in range(8):  
            		tMean, tVar = [],[]

			# perforimg wavelet decomposition of the region using db1 mode at level 1
	    		coeffs=pywt.wavedec2(imArray[r*cY:r*cY+cY,c*cX:c*cX+cX], 'db1', 1)

			# finding the mean and variance of 4 arrays that resulted from decomposition
			# coeffs[0] will be a low res copy of image region
			# coeffs[1][0..2] will be 3 band passed filter results in horizontal, 
			# vertical and diagonal directions respectively 
           		for i in range(4):

			    # appending the mean and variance values of a region into two vectors
			    if i == 0:   
		    		tMean.append(np.mean(coeffs[i]))
                    		tVar.append(np.var(coeffs[i]))
			    else:
		    		tMean.append(np.mean(coeffs[1][i-1]))
                    		tVar.append(np.var(coeffs[1][i-1]))

			# appending the mean and variance vectors of all regions along the row 
	    		imRMean.append(tMean)
            		imRVar.append(tVar)

		    # appending the mean and variance vectors of all rows
    		    imMean.append(imRMean)
    		    imVar.append(imRVar)
		
		# appending mean and variance vectors into one features vector
		feats.append(imMean)
		feats.append(imVar)

		# flattening the features vector
		feats = np.asarray(feats)
		feats =  cv2.normalize(feats,feats)
		feats = feats.flatten()

		# returning the features vector / histogram
		return feats

class ColorTree:

	def __init__(self, bins):
		# saving the bin count for our 3D histogram
		self.bins = bins

	def color_tree(self, img):
		# converting the given image to grayscale and initializing
		# an array to save features used to represent the image
		img = img/42
		img = img*42
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		feats = []
 
		# calculating the center of the image
		(h, w) = img.shape[:2]
		(cX, cY) = (int(w * 0.5), int(h * 0.5))

		# partitioning the image into four rectangles/segments (top-left,
		# top-right, bottom-right, bottom-left)
		segs = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
			(0, cX, cY, h)]
 
		# building a mask in ellipse shape to represent the image centre
		(Xaxis, Yaxis) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
		ellipMask = np.zeros(img.shape[:2], dtype = "uint8")
		cv2.ellipse(ellipMask, (cX, cY), (Xaxis, Yaxis), 0, 0, 360, 255, -1)
 
		# iterating over the segments
		for (startX, endX, startY, endY) in segs:
			# building a mask for each image corner, subtracting the elliptical center
			cornerMask = np.zeros(img.shape, dtype = "uint8")
			cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
			cornerMask = cv2.subtract(cornerMask, ellipMask)
 
			# extracting an histogram of colors from the corner regions, then updating the
			# features array
			hist = self.histogram(img, cornerMask)
			feats.extend(hist)
 
		# extracting an histogram from the elliptical centre, then updating the
		# features array
		hist = self.histogram(img, ellipMask)
		hist.sort
		feats.extend(hist)
 
		# returning the array with features
		return feats

	def histogram(self, img, mask):
		# getting a 3D histogram from the masked part of the image
		# using the given bin count per channel, followed by histogram normalization
		hstgm = cv2.calcHist([img], [0], mask, self.bins,[0, 253])
		hstgm =  cv2.normalize(hstgm,hstgm)
		hstgm = hstgm.flatten()
 
		# returning the histogram
		return hstgm

class DescribeColor:

	def __init__(self, bins):
		# saving the bin count for our 3D color histogram
		self.bins = bins
	
	def describe_color(self, img):
		# converting the color space of the image from BGR to HSV and initializing
		# an array to save features used to represent the image
		img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		feats = []
 
		# calculating the center of the image
		(h, w) = img.shape[:2]

		(cX, cY) = (int(w * 0.5), int(h * 0.5))
		# partitioning the image into four rectangles/segments (top-left,
		# top-right, bottom-right, bottom-left)
		segs = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
			(0, cX, cY, h)]
 
		# building a mask in ellipse shape to represent the image centre
		(Xaxis, Yaxis) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
		ellipMask = np.zeros(img.shape[:2], dtype = "uint8")
		cv2.ellipse(ellipMask, (cX, cY), (Xaxis, Yaxis), 0, 0, 360, 255, -1)
 
		# iterating over the segments
		for (startX, endX, startY, endY) in segs:
			# building a mask for each image corner, subtracting the elliptical center
			cornerMask = np.zeros(img.shape[:2], dtype = "uint8")
			cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
			cornerMask = cv2.subtract(cornerMask, ellipMask)
 
			# extracting an histogram of colors from the corner regions, then updating the
			# features array
			hist = self.histogram(img, cornerMask)
			feats.extend(hist)
 
		# extracting an histogram of colors from the elliptical centre, then updating the
		# features array
		hist = self.histogram(img, ellipMask)
		feats.extend(hist)
 
		# returning the array with features
		return feats

	def histogram(self, img, mask):
		# getting a 3D color histogram from the masked part of the image
		# using the given bin count per channel, followed by histogram normalization
		hstgm = cv2.calcHist([img], [0, 1, 2], mask, self.bins,
			[0, 180, 0, 256, 0, 256])
                hstgm =  cv2.normalize(hstgm,hstgm)
		hstgm = hstgm.flatten()
 
		# returning the histogram
		return hstgm
