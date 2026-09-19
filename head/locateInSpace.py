# takes care of both the robot's spacial awareness relative to itself, and where the object is located
# also takes care of finding obstacles and stuff, this is where the yolo thingies should go 
# need to have fixed constants class for the sizings of objects, to accurately tell distance? (ex. hands, doors)
# should average the depth data from the center of the desired object, vs the optical conversions

import cv2

capture = cv2.VideoCapture(0)

success, frame = capture.read()

if success:
    cv2.imwrite( "test.png", frame )
    print( "saved sample" )