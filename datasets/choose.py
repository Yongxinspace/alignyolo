import os
import cv2

with open('test_visible2.txt','r') as f:
     for ix,line  in enumerate(f):
         line = line.strip('\n')
         files = line.split(' ')
         img_path= files[0]
         f = os.path.exists(img_path)
         if f == False:   
              print(img_path)
         else:
            img = cv2.imread(img_path)
            try :
	            img.shape
            except:
                    print(img_path)
