
pip install opencv-python
import cv2
img = cv2.imread(&#39;C:/Users/Falguni Bharadwaj/Pictures/with_mask/with_mask_1.jpg&#39;)
img.shape
img[0]
while True:
cv2.imshow(&#39;result&#39;,img)
#27-ASCII of Escape
if cv2.waitKey(2) == 27:
break
cv2.destroyAllWindows()
capture = cv2.VideoCapture(0) #to initialize camera
data = [] #to store face data
while True:
flag, img = capture.read() #read video fram by frame and return true/false and one frame at a time
if flag: #will check if flag is True (whether camera is available or not)

faces = haar_data.detectMultiScale(img) #detecting face from the frame
for x,y,w,h in faces: #fetching x,y,w,h of face detected in frame
cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 255), 4) #drawing rectangle on face
face = img[y:y+h, x:x+w, :] #slicing only face from the frame
face = cv2.resize(face, (50,50)) #resizing all afces to 50x50 so that all images are of same size
print(len(data))
if len(data) &lt; 200: #condition for only storing 200 images
data.append(face) #storing face data
cv2.imshow(&#39;result&#39;,img) #to show the window
#27-ASCII of Escape
if cv2.waitKey(2) == 27 or len(data &gt;= 200): #break loop if escape is pressed or 200 faces are
stored
break

capture.release() #release the camera object held by opencv
cv2.destroyAllWindows() #close all the windows opened by opencv
np.save(&#39;without_mask.npy&#39;, data)
np.save(&#39;with_mask.npy&#39;, data)
plt.imshow(data[0])
import numpy as np
import cv2
with_mask = np.load(&#39;with_mask.npy&#39;)
without_mask = np.load(&#39;without_mask.npy&#39;)
with_mask.shape
without_mask.shape
X = np.r_[with_mask, without_mask]

X.shape
labels = np.zeros(X.shape[0])
labels[200:] = 1.0
names = {0 : &#39;Mask&#39;, 1 : &#39;No Mask&#39;}
#SVM - Support Vector Machine
#SVC - Support Vector Classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,labels, test_size = 0.25)
x_train.shape
# X_test = pca.transform(x_test)
y_pred = svm.predict(x_test)
accuracy_score(y_test, y_pred)
haar_data = cv2.CascadeClassifier(&#39;haarcascade_frontalface_default.xml&#39;)
capture = cv2.VideoCapture(0)
data = []
font = cv2.FONT_HERSHEY_COMPLEX
while True:
flag, img = capture.read()
if flag:
faces = haar_data.detectMultiScale(img)
for x,y,w,h in faces:
cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 4)
face = img[y:y+h, x:x+w, :]
face = cv2.resize(face, (50,50))

face = face.reshape(1.-1)
pred = svm.predict(face)[0]
n = name[int(pred)]
cv2.putText(img, n, (x,y), font, 1, (244,250,250), 2)
print(n)
cv2.imshow(&#39;result&#39;,img)
#27-ASCII of Escape
if cv2.waitKey(2) == 27:
break
capture.release()
cv2.destoryAllWindows()
