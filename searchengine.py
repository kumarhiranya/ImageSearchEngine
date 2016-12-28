import cv2
import numpy as np
from matplotlib import pyplot
import glob
import os
import cPickle


class rgbhist:
    def __init__(self, bins):
        self.bins = bins

    def describe(self, image):
        hist = cv2.calcHist([image], [0, 1, 2], None, self.bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist)
        return hist.flatten()


class Searcher:
    def __init__(self, index):
        # store index of images
        self.index = index

    def search(self, queryFeatures):
        # initialize dictionary of results
        results = {}

        # loop over the index
        for (k, features) in self.index.items():
            d = self.chi2_distance(features, queryFeatures)
            results[k] = d

        # sort the results based on distances
        results = sorted([(v, k) for (k, v) in results.items()])

        # return results
        return results

    def chi2_distance(self, histA, histB, eps=1e-10):
        # compute and return the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                          for (a, b) in zip(histA, histB)])
        return d


def check(names):
    #check if the given list of filenames is the same as filenames.txt, i.e. check if the directory has been altered
    f = open(dir + "/filenames.txt", "r")
    prevfiles = cPickle.load(f)
    if sorted(prevfiles) == sorted(names):
        return False
    else:
        return True
    f.close()


dir = "C:\Users\Hiranya\Desktop\imageproc"
imgdir= "image data base path"
query = "query image path"
qimage = cv2.imread(query)
w, h, c = qimage.shape
qimage = cv2.resize(qimage, (h / 2, w / 2))
descriptor = rgbhist([8, 8, 8])
index = {}
names = []
i=0
for name in glob.glob(imgdir + "/*.jpg"):    # ONLY considers .jpg images
        names.insert(i, os.path.basename(name))

if check(names):                             # Check if the directory has been modified by comparing the list of filenames in the dir
    print "Directory altered! Updating filenames.txt..."
    f1 = open(dir + "/filenames.txt", "w")   # If modified, update the filenames.txt
    f1.write(cPickle.dumps(names))
    f1.close()
    print "filenames.txt updated, Updating index..."
    f = open(dir + "/index.txt", "w")         # Update the index by iterating thru the images and getting their flattened histograms 
    for name in glob.glob(imgdir + "/*.jpg"):
        img = cv2.imread(name)
        k = os.path.basename(name)
        index[k] = descriptor.describe(img)
    f.write(cPickle.dumps(index))             # update index.txt
    f.close()
    print "Index Updated!"


else:
     print "Directory not altered! Loading Index from index.txt"
     f = open(dir + "/index.txt", "r")         # If directory not altered, load the index from index.txt
     index = cPickle.load(f)
     f.close()

engine = Searcher(index)                       # Initialize engine of searcher class with the index
results = engine.search(descriptor.describe(qimage))    #search for the query image using its flattened histogram

canvas = np.zeros((1080, 1920, 3), dtype="uint8")   # create a "canvas" for the montage containing results
i = 0
res = []                    #list containing the filenames of the results in decending order of similarity
cv2.imshow("Query image", qimage)       #show query image

for a, b in results:           # create a montage of the results
    if i < 6:
        res.insert(i, b)            
        if i < 3:               # first 3 images in the first row

            z = i * 600
            temp = cv2.imread(imgdir + "/" + b)
            temp = cv2.resize(temp, (600, 490))
            canvas[0:490, z: z + 600] = temp
            
        else:                   # next 3 images in the second row 
            z = (i - 3) * 600
            temp = cv2.imread(imgdir + "/" + b)
            temp = cv2.resize(temp, (600, 490))
            canvas[490:980, z: z + 600] = temp
            
    i += 1
cv2.imshow("canvas", canvas)

cv2.waitKey()
cv2.destroyAllWindows()
