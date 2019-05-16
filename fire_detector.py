
# Target: Detect fire in the CCTV video 

# Approachs: 1. Segregation with color (BGR: R > other); HSI: ( inRange); YCrCb (illumination)
#              2. Dynamic nature of flame ( Motion detection) 
#     Finally: combine using SVM

# Problems: 1. Feature selection 
#           2. Illumination changes
#           3. Same color objects in image (in case of color segmentation methods)
#           4. Stable flame
 
# Comments: High intensity around flame region can be used for detections
# Alternate video source: https://www.youtube.com/watch?v=LEaZKCAy4_8 (chawkbazar)


"""
    System Requirements:
        Python (ver 3)
        Opencv (ver >=3)
        Pandas

    To run the program simply execute : python fire_detector.py

"""


import cv2
import numpy as np
import pandas as pd
import argparse
import pickle


# feature extraction for SVM model
def fd_hu_moments(gray):
    feature = cv2.HuMoments(cv2.moments(gray)).flatten()
    return feature

# Creating dataset from the filtered contour regions
def label_contour(frame):
    try:
        frame = cv2.resize(frame, (128, 128))
    except:
        pass
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    feature = fd_hu_moments(gray_frame)
    cv2.imshow("contour window", frame)
    
    ret = False # Flag to stop the process
    
    # key_input = input('label ....')
    label = 0

    if cv2.waitKey(0) == ord('q'):
        ret = True # Stop the dataset record
    elif cv2.waitKey(0) == ord('y'):
        label = 1
    else:
        label = 0
    
    print("****** {} *************".format(label))
    return ret, feature, label


def red_region_rgb(frame, gray_frame, RTH=220):

    b,g,r = cv2.split(frame)
    res = np.zeros(frame.shape[:2], dtype=np.uint8)

    # to reduce the effect of illumination
    # B = cv2.equalizeHist(b)
    # G = cv2.equalizeHist(g)
    # R = cv2.equalizeHist(r)
    
    # individual threshold
    B = b
    G = g 
    R = r

    Roi = (np.logical_and(np.logical_and(R > G, G > B), R > RTH))
    res[Roi] = gray_frame[Roi]
    _, res = cv2.threshold(res, 50, 255, cv2.THRESH_BINARY)

    return res


# YCrCb color space better distinguish luminance and chrominance information of the image
def red_region_yCrCb(frame, gray_frame):

    frame_y = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    res = np.zeros(frame.shape[:2], dtype=np.uint8)

    Y = frame_y[:,:,0]
    Cr = frame_y[:,:,1]
    Cb = frame_y[:,:,2]

    Roi = (np.logical_and(Y > Cr, Cr > Cb))
    res[Roi] = gray_frame[Roi]
    _, res = cv2.threshold(res, 50, 255, cv2.THRESH_BINARY)

    return res


# combining the detect red regions from both color spaces
def red_region(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res1 = red_region_rgb(frame, gray_frame)

    res2 = red_region_yCrCb(frame, gray_frame)

    res = cv2.bitwise_or(res1, res2)

    return res



""" main """
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', type = bool, default=False, help="set if in training mode else False to test")
    parser.add_argument('-v','--videofile', default="sample.mp4", help="path to video file")
    arg = parser.parse_args()

    cl_flag = arg.train # Set if train else False to load model and predict

    filename = arg.videofile
    cap = cv2.VideoCapture(filename)
    # total_frame = cap.get(7)
    # cap.set(100, total_frame//2)

    last_frame = -1 # no last frame

    feature_set = []
    label_set = []

    filename_cls = 'flame_model.sav'
    file_sc = 'ftr_scale.sav'

    if not cl_flag:
        classifier = pickle.load(open(filename_cls, 'rb'))
        scalar = pickle.load(open(file_sc, 'rb'))

    while(cap.isOpened()):
        ret_frame, frame = cap.read()
        try:
            frame = cv2.resize(frame, (640, 480))
        except:
            continue

        # frame = frame[160:300, 260:420]

        kernel = np.ones((7, 7), dtype=np.uint8) # for morphological operations

        mr = 4 # pixel margin

        """ Red region detection """
        # res2 = red_region_yCrCb(frame)
        res2 = red_region(frame)
        res2 = cv2.morphologyEx(res2, cv2.MORPH_OPEN, kernel)
        cv2.imshow("red region", res2)


        """ Applying frame differencing(to capture the dynamic changes) """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = np.zeros(gray.shape, np.uint8)
        # gray_frame = cv2.equalizeHist(gray_frame)
        # gray_frame = frame[:,:,2] # motion of flames is not captured efficiently so capturing changes over R component
        gray_frame[res2 > 1] = gray[res2 > 1]

        dilute_frame = cv2.GaussianBlur(gray_frame, (9,9), 20) 

        if np.all(last_frame == -1): # recording the first frame
            last_frame = dilute_frame
            continue

        # absolute difference between the frame to detect the changed (moving) region
        delta_frame = cv2.absdiff(dilute_frame, last_frame) 
        
        # Binarize
        _, bin_frame = cv2.threshold(delta_frame, 50, 255, cv2.THRESH_BINARY)

        # filling holes -> closing
        close_res = cv2.morphologyEx(bin_frame, cv2.MORPH_CLOSE, kernel)

        # updating the frame
        last_frame = dilute_frame
        frame_copy = frame.copy()

        res = close_res
        cv2.imshow("motion", res)


        """ Combined result """
        res_com = cv2.bitwise_and(res, res2)
        res_com = cv2.morphologyEx(res_com, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(res_com, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]


        ret = False
        for contour in cnts: # approximating the closed curve
            # if cv2.contourArea(contour) < 8:
            #     continue

            frame_copy = frame.copy() # freshly drawing on each curve

            appr_curve = contour
            # peri = cv2.arcLength(contour, True) # not improving performance 
            # appr_curve = cv2.approxPolyDP(contour, 0.02 * peri, closed = True)
            (x, y, w, h) = cv2.boundingRect(appr_curve)

            lf_x = x if x-mr < 0 else x-mr
            lf_y = y if y-mr < 0 else y-mr

            
            
            if cl_flag: # training
                
                cv2.rectangle(frame_copy, (lf_x, lf_y), (x+w+mr, y+h+mr), (0, 255, 0), 2)
                cv2.imshow("com", frame_copy)
                try:
                    # extracting featurees and creating data set 
                    ret, fd, lb = label_contour(frame[lf_y:y+h+mr, lf_x:x+w+mr])
                    
                    if ret:
                        break

                    # feature_set = []
                    feature_set.append(np.hstack((fd, lb)))
                    # feature_set.append(lb)
                    label_set.append(lb)
                    
                    print("label vector", label_set)
                except:
                    continue
            else:
                """ 
                    The implimentation is remaining 
                    It is expected to remove false positive cases and improve the performance drastically
                """
                # predicting if correct using SVM
                feature = fd_hu_moments(gray[lf_y:y+h+mr, lf_x:x+w+mr])
                feature = np.asarray(feature).reshape((1,-1))
                feature_sc = scalar.transform(feature)
                y_res = classifier.predict(feature_sc)

                if y_res == 1: #fire not predicted
                    cv2.rectangle(frame_copy, (lf_x, lf_y), (x+w+mr, y+h+mr), (0, 255, 0), 2)
                    # frame_copy = frame.copy() # freshly drawing on each curve
                    # continue

            
            if not cl_flag: # testing
            # drawing the bounding box on ROI
                cv2.imshow("com", frame_copy)            


        # res_com = np.hstack((gray_frame, res_com))
        # cv2.imshow("com", res_com)

        if cv2.waitKey(30) == ord('q') or not(ret_frame) or ret:
            # dataset = {'feature': feature_set, 'label': label_set}
            if cl_flag: 
                dataset = feature_set
                df = pd.DataFrame(dataset)
                df.to_csv('flame.csv', index=False)
            break


    cap.release()
    cv2.destroyAllWindows()