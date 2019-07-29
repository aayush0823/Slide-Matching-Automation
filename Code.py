import os
import random
import cv2
import sys
import threading
import numpy as np
from skimage.measure import compare_ssim as ssim
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import difflib
import Levenshtein
dir1=sys.argv[2]
dir2=sys.argv[1]
f = open("20171203_20171188_20171033.txt","w")
#wrg = open("wrongtillnow","w")
#wrong = 0
threads = []
topfiles = 6
def read_images():
    i = 0
    global wrong,f,dir1,dir2,threads
    for filename1 in os.listdir(dir1):
        img1 = cv2.imread(os.path.join(dir1,filename1))
        maxi=-1
        matchallframes(img1,filename1,[])
    #print("Number Of Wrongs",wrong)
def correlation_coefficient(im, im2):
    if im.shape != im2.shape:
        new_dims = (im.shape[1],im.shape[0])
        im2 = cv2.resize(im2,new_dims)
    product = np.mean((im - im.mean()) * (im2 - im2.mean()))
    stdev = im.std() * im2.std()
    if stdev == 0:
        return 0
    else:
        product /= stdev
        return product
def matchallframes(img1,filename1,setpts):
    global wrong,f,dir1,dir2
    max_filename=""
    path, dirs, files = next(os.walk(dir2))
    file_count = len(files)
    if file_count > 6:
        topfiles = 6
    else:
        topfiles = file_count
    for filename2 in os.listdir(dir2):
        img2 = cv2.imread(os.path.join(dir2,filename2))
        temp = correlation_coefficient(img1,img2)
        setpts.append([temp,filename2])
        #print("here",filename2)
    setpts.sort(key = lambda x:x[0],reverse = True)
    #print(setpts)
    vals = []
    sssvals = []
    distmap = []
    orig = pytesseract.image_to_string(Image.open(os.path.join(dir1,filename1)))
    for i in range(0,topfiles):
        im3 = cv2.imread(os.path.join(dir2,setpts[i][1]))
        temp = cmpimage(img1,im3)
        vals.append(temp)
        temp2 = pytesseract.image_to_string(Image.open(os.path.join(dir2,setpts[i][1])))
        distmap.append(Levenshtein.ratio(orig,temp2))
        #res = cv2.matchTemplate(img1,im3,cv2.TM_CCORR_NORMED)
        #res = cv2.norm(np.subtract(img1,im3),cv2.NORM_L2)
        #res = np.sqrt(np.mean((np.subtract(img1,im3))**2))
        #res = signal.correlate2d(im3,img1, boundary='symm', mode='same')
        res = ssim(img1,im3,multichannel=True)
        sssvals.append(res)
        #res = np.sum(np.absolute(img1 - im3)) / (im3.shape[0]*im3.shape[1]) / 255
        #print(vals[i],setpts[i][1],res)
    sfli = sssvals.index(max(sssvals))
    fli = vals.index(max(vals))
    ocrind = distmap.index(min(distmap))
    vals.sort(reverse = True)
    #print(sfli,fli)
    if(sfli == fli):
        max_filename = setpts[fli][1]
    else:
        if abs(vals[0]-vals[1]) < 8:
            max_filename = setpts[ocrind][1]
        max_filename = setpts[fli][1]
    #print("rvvrevrvvrt4e",str(max_filename))
    name1 = str(filename1)
    name2 = str(max_filename)
    #print(name1 + ' ' + name2 + '\n')
    f.write(name1 + ' ' + name2 + '\n')
    #if(name1[0:5] != name2[0:5]):
        #print("Wrong Here")
        #wrg.write(name1 + ' ' + name2 + '\n')
        #wrong+=1
def cmpimage(img1,img2):
	sift_output = cv2.xfeatures2d.SIFT_create()
	result_dict = dict()
	points_1, values = sift_output.detectAndCompute(img1, None)
	ind_p = dict(algorithm=0, trees=5)
	points_2, values_2 = sift_output.detectAndCompute(img2, None)
	flann = cv2.FlannBasedMatcher(ind_p, result_dict)	 
	goodpts = []
	matches = flann.knnMatch(values, values_2, k=2)
	ratio = 0.565
	for m, n in matches:
	    if m.distance < ratio*n.distance:
	        goodpts.append(m)
	#print(len(goodpts))
	return len(goodpts)
if __name__ == "__main__":
	read_images()
