# importing the needed bindings
import numpy as np
import csv
 
class Ranker:

	def __init__(self, hsv, texture, tree):
		# saving the given index paths for hsv, tree and texture feature files
		self.hsv = hsv
		self.tree = tree
		self.texture = texture

	def rank(self, queryFeats, queryTexture, queryTree, limit = 15):
		# initializing dictionaries to save results of color, texture
		# and tree matching
		finresult, cresult, txresult, tresult = {}, {}, {}, {}

		with open(self.hsv) as p:
			# opening the index path for reading
			read = csv.reader(p)
 
			# iterating over the records in the index
			for record in read:
				# spliting out the image ID and features
				cfeats = [float(x) for x in record[1:]]
				# calculating chi-squared distance between the index features and query features
				cdis = self.chi_sqrd_distance(cfeats, queryFeats)
 
				# with image ID as key and distance as value, udpating the result dictionary
				cresult[record[0]] = cdis
				
			# closing the reader
			p.close()

		with open(self.texture) as p:
			# opening the index path for reading
			read = csv.reader(p)
 
			# iterating over the records in the index
			for record in read:
				# spliting out the image ID and features
				txfeats = [float(x) for x in record[1:]]
				# calculating chi-squared distance between the index features and query features
				txdis = self.chi_sqrd_distance(txfeats, queryTexture)
 				
				# with image ID as key and distance as value, udpating the result dictionary
				txresult[record[0]] = txdis
				
			# closing the reader
			p.close()

		with open(self.tree) as p:
			# opening the index path for reading
			read = csv.reader(p)
 
			# iterating over the records in the index
			for record in read:
				# spliting out the image ID and features
				tfeats = [float(x) for x in record[1:]]
				# calculating chi-squared distance between the index features and query features
				tdis = self.chi_sqrd_distance(tfeats, queryTree)
 				
				# with image ID as key and distance as value, udpating the result dictionary
				tresult[record[0]] = tdis
 
			# closing the reader
			p.close()

		# iterating over all images in all results, combining their ranks from 
		# color, texture and tree comparisons
		for (k, v) in cresult.items():
			# getting a weighted sum of ranks for the image
			finresult[k] = (cresult[k]*0.90) + (tresult[k]*0.05) + (txresult[k]*0.05)
 
		# sorting our results, in the increasing order of distances
		finresult = sorted([(v, k) for (k, v) in finresult.items()])
 
		# returning our results within the specified limit
		return finresult[:limit]

	def chi_sqrd_distance(self, histA, histB, eps = 1e-10):
		# calculating the chi-squared distance
		dist = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])
 
		# returning the calculated chi-squared distance
		return dist
