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
        # store our index of images
        self.index = index

    def search(self, queryFeatures):
        # initialize our dictionary of results
        results = {}

        # loop over the index
        for (k, features) in self.index.items():
            d = self.chi2_distance(features, queryFeatures)

            # now that we have the distance between the two feature
            # vectors, we can udpate the results dictionary -- the
            # key is the current image ID in the index and the
            # value is the distance we just computed, representing
            # how 'similar' the image in the index is to our query
            results[k] = d

        # sort our results, so that the smaller distances (i.e. the
        # more relevant images are at the front of the list)
        results = sorted([(v, k) for (k, v) in results.items()])

        # return our results
        return results

    def chi2_distance(self, histA, histB, eps=1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                          for (a, b) in zip(histA, histB)])

        # return the chi-squared distance
        return d


def check(names):
    f = open(dir + "/filenames.txt", "r")
    prevfiles = cPickle.load(f)
    if sorted(prevfiles) == sorted(names):
        return False
    else:
        return True
    f.close()


dir = "C:\Users\Hiranya\Desktop\imageproc"
imgdir= "A:\eBooks and Manga\Blade play\Blade Play v01"
query = "A:\eBooks and Manga\Blade play\Blade Play v01\Blade Play! 01_0002.jpg"
qimage = cv2.imread(query)
w, h, c = qimage.shape
qimage = cv2.resize(qimage, (h / 2, w / 2))
descriptor = rgbhist([8, 8, 8])
index = {}
names = []
i=0
for name in glob.glob(imgdir + "/*.jpg"):
        names.insert(i, os.path.basename(name))

if check(names):
    print "Directory altered! Updating filenames.txt..."
    f1 = open(dir + "/filenames.txt", "w")
    f1.write(cPickle.dumps(names))
    f1.close()
    print "filenames.txt updated, Updating index..."
    f = open(dir + "/index.txt", "w")
    for name in glob.glob(imgdir + "/*.jpg"):
        img = cv2.imread(name)
        k = os.path.basename(name)
        index[k] = descriptor.describe(img)
    f.write(cPickle.dumps(index))
    f.close()
    print "Index Updated!"


else:
     print "Directory not altered! Loading Index from index.txt"
     f = open(dir + "/index.txt", "r")
     index = cPickle.load(f)
     f.close()

engine = Searcher(index)
results = engine.search(descriptor.describe(qimage))

canvas = np.zeros((1080, 1920, 3), dtype="uint8")
i = 0
res = []
cv2.imshow("Query image", qimage)

for a, b in results:
    if i < 6:
        res.insert(i, b)
        if i < 3:

            z = i * 600
            temp = cv2.imread(imgdir + "/" + b)
            temp = cv2.resize(temp, (600, 490))
            # print i, z, temp.shape
            canvas[0:490, z: z + 600] = temp
            # cv2.rectangle(roi[i],(0,0),(600,600), (0,255,0), 5)
        else:
            z = (i - 3) * 600
            temp = cv2.imread(imgdir + "/" + b)
            temp = cv2.resize(temp, (600, 490))
            # cv2.rectangle(roi[i], (0, 0), (600, 600), (0, 255, 0), 5)
            canvas[490:980, z: z + 600] = temp
            # print i, z, temp.shape
    i += 1
#print index, res
cv2.imshow("canvas", canvas)

cv2.waitKey()
cv2.destroyAllWindows()