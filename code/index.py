# importing the needed bindings
from descriptor import DescribeColor
from descriptor import DescribeTexture
from descriptor import ColorTree
import cv2
import argparse
import glob

# instantiating the classes for color, texture and tree description
cdes = DescribeColor((16, 32, 1))
txdes = DescribeTexture()
tdes = ColorTree([6])

# building the argument parser and parse the command line arguments
argprse = argparse.ArgumentParser()
argprse.add_argument("-d", "--dataset", required = True,
	help = "FilePath to the folder that has target images to be indexed")
argprse.add_argument("-c", "--hsv", required = True,
	help = "FilePath where the computed hsv index is to be saved")
argprse.add_argument("-t", "--texture", required = True,
	help = "FilePath where the computed texture index is to be saved")
argprse.add_argument("-b", "--btree", required = True,
	help = "FilePath where the computed tree index is to be saved")
argmnts = vars(argprse.parse_args())
 
# opening the respective output index files in write mode
clroutput = open(argmnts["hsv"], "w")
treeoutput = open(argmnts["btree"], "w")
texoutput = open(argmnts["texture"], "w")
 
# using glob to capture the paths of the images and iterate through them
for path in glob.glob(argmnts["dataset"] + "/*.jpg"):
	# getting the unique filenames from the image path
	imgID = path[path.rfind("/") + 1:]
	# loading the image
	image = cv2.imread(path)

	# getting the color, texture and tree features from the image
	color = cdes.describe_color(image)
	tree = tdes.color_tree(image)
	texture = txdes.describe_texture(image)

	# writing the color features to the clroutput file
	color = [str(f) for f in color]
	clroutput.write("%s,%s\n" % (imgID, ",".join(color)))
 
	# writing the tree features to the treeoutput file
	tree = [str(t) for t in tree]
	treeoutput.write("%s,%s\n" % (imgID, ",".join(tree)))

	# writing the texture features to the texoutput file
	texture = [str(tx) for tx in texture]
	texoutput.write("%s,%s\n" % (imgID, ",".join(texture)))
 
# close all the index files
clroutput.close()	
treeoutput.close()	
texoutput.close()	
