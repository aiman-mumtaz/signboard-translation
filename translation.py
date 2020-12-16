import cv2
import PIL
import numpy as np
from PIL import Image
import pytesseract
from scipy.ndimage import interpolation as inter
from textblob import TextBlob


pytesseract.pytesseract.tesseract_cmd = "C:\\Tesseract\\tesseract.exe"


# Load image
image = cv2.imread("./images/19.jpg")


# 1. Image Preprocessing

# a. skew correction

delta = 0.05
limit = 5

def determine_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    histogram = np.sum(data, axis=1)
    score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
    return histogram, score

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

scores = []
angles = np.arange(-limit, limit + delta, delta)
for angle in angles:
    histogram, score = determine_score(thresh, angle)
    scores.append(score)

best_angle = angles[scores.index(max(scores))]

(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)



# ----------------------------------------------------------------------------------------------------------------------------------------
# Image = cv2.bitwise_not(image)

# # Threshold and invert
# _,thr = cv2.threshold(rotated,127,255,cv2.THRESH_BINARY)
# inv   = 255 - thr

# # Perform morphological closing with square 7x7 structuring element to remove details and thin lines
# SE = np.ones((7,7),np.uint8)
# closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, SE)

# # Find row numbers of dark rows
# meanByRow=np.mean(closed,axis=1)
# rows = np.where(meanByRow<50)

# # Replace selected rows with those from the inverted image
# rotated[rows]=inv[rows]

# ---------------------------------------------------------------------------------------------------------------------------------------------



# FOOL PROOF METHODS    
# b. Noise removal from the image
clean = cv2.fastNlMeansDenoising(image) 
# ------------------------------------------------------------------------------------------------------------------------------------------------------------
# c. Thresholding
img = cv2.cvtColor(clean,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  
# thresh = cv2.bitwise_not(thresh)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#2. Text Detection

text = pytesseract.image_to_string(thresh,lang='hin',config= '--psm 6')
hi_blob = TextBlob(text)




# 3. Translation
result = hi_blob.translate(to='en')

print(result)


# 4. Post Processing

# Resulting image
cv2.imshow('Image',thresh)
cv2.waitKey(0)