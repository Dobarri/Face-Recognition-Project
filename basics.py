import cv2 as cv
import face_recognition

#load image
imgIU = face_recognition.load_image_file('imagebasic/iu1.jpg')
imgIU = cv.cvtColor(imgIU,cv.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('imagebasic/elon musk.jpg')
imgtest = cv.cvtColor(imgtest,cv.COLOR_BGR2RGB)

#location
faceLoc = face_recognition.face_locations(imgIU)[0]
encodeIU = face_recognition.face_encodings(imgIU)[0]
print(faceLoc)
cv.rectangle(imgIU,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLoc_t = face_recognition.face_locations(imgtest)[0]
encode_t = face_recognition.face_encodings(imgtest)[0]
print(faceLoc_t)
cv.rectangle(imgtest,(faceLoc_t[3],faceLoc_t[0]),(faceLoc_t[1],faceLoc_t[2]),(255,0,255),2)

#result
results = face_recognition.compare_faces([encodeIU],encode_t)
#distance
faceDis = face_recognition.face_distance([encodeIU],encode_t)
print(results, faceDis)
#put text
cv.putText(imgtest,f'{results} {round(faceDis[0],3)}',(50,50),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv.imshow('IU', imgIU)
cv.imshow('imgtest', imgtest)
cv.waitKey(0)