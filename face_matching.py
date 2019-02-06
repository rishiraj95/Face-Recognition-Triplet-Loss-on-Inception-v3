import os
from numpy import linalg as LA
import numpy as np
import cv2
from numpy import linalg as LA
import numpy as np
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage

def grab_face_vector():
    app = ClarifaiApp(api_key='df477c72fbb246b99921528fa25c54bd')
    model = app.models.get("d02b4508df58432fbb84e800597b8959")
    cap=cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        cv2.imshow('camera',frame)
        key=cv2.waitKey(1) & 0xFF
        if key==13:
            image_name="face.jpg"
            cv2.imwrite(image_name,frame)
            break

    cap.release()
    cv2.destroyAllWindows()
    image = ClImage(file_obj=open(image_name, 'rb'))
    result=model.predict([image])
    outputs=result['outputs'][0]
    data=np.array(outputs['data']['regions'][0]['data']['embeddings'][0]['vector'])
    image=None
    os.remove(image_name)
    return data

inp1=input("Please enter '1' to take the first picture")

if inp1=='1':
    data1=grab_face_vector()
else:
    print("Wrong input...Try again")



if inp1=='1':
    inp2=input("Please enter '2' to compare new pictures to the first picture")
    if inp2=='2':
        data2=grab_face_vector()
        if LA.norm(data1-data2)<=0.85:
            print("Match")
        else:
            print("Not a Match")
    else:
        print("Wrong input...Try again")

    
