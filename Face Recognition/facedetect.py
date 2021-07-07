import numpy as np
import pandas as pd
import cv2
f_name = 'face_data.csv'
dataset = pd.read_csv(f_name)

X=dataset.iloc[:,1:-1].values
Y= dataset.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler

sc= StandardScaler()
X_scaled = sc.fit_transform(X)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_scaled,Y)

cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
f_list = []

while True:
    ret,frame= cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray,1.5,5)
    X_test = []
    for face in faces:
        x,y,w,h = face
        im_face = gray[y:y+h,x:x+w]
        im_face = cv2.resize(im_face,(100,100))
        im_face = im_face.reshape(-1)
        
        X_test.append(im_face)
    
    if len(faces)>0:
        Y_pred = classifier.predict(np.array(sc.transform(X_test)))
        
        
        for i,face in enumerate(faces):
            x,y,w,h = face
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),3)
            cv2.putText(frame, Y_pred[i], (x-50,y-50), cv2.FONT_HERSHEY_DUPLEX, 2, (255,0,0),2)
            
    cv2.imshow("Result",frame)
    key=cv2.waitKey(1)
    
    if key & 0xFF == ord('q') :
        break
    
cap.release()
cv2.destroyAllWindows()
