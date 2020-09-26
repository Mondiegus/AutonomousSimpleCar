import numpy as np
import cv2
import time
import mss
import pandas as pd
import keyboard
import os
import tensorflow as tf
from numba import jit
import threading
import time

class myThreadStraight (threading.Thread):
   def __init__(self, param):
      threading.Thread.__init__(self)
      self.param = param
   def run(self):
      # Get lock to synchronize threads
      threadLock.acquire()
      goStraight(self.param)
      # Free lock to release next thread
      threadLock.release()

class myThreadLeft (threading.Thread):
   def __init__(self, param1, param2):
      threading.Thread.__init__(self)
      self.param1 = param1
      self.param2 = param2
   def run(self):
      # Get lock to synchronizae threads
      threadLock.acquire()
      goLeft(self.param1, self.param2)
      # Free lock to release next thread
      threadLock.release()

class myThreadRight (threading.Thread):
   def __init__(self, param1, param2):
      threading.Thread.__init__(self)
      self.param1 = param1
      self.param2 = param2
   def run(self):
      # Get lock to synchronize threads
      threadLock.acquire()
      goRight(self.param1, self.param2)
      # Free lock to release next thread
      threadLock.release()


def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    try:
        lane_lines = []
        height, width = frame.shape
        left_fit = []
        right_fit = []
        Ys = []
        cords = []
        ml = 0
        mr = 0
        boundary = 1 / 2
        left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
        right_region_boundary = width * boundary  # right lane line segment should be on right 1/3 of the screen
        for line_segment in line_segments:
            for x1, y1, x2, y2 in line_segment:
                if x1 == x2:
                    continue
                Ys += [y1, y2]
                min_y = min(Ys)
                max_y = 700
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if slope < 0:
                    if x1 < left_region_boundary and x2 < left_region_boundary:
                        left_fit.append((slope, intercept))
                else:
                    if x1 > right_region_boundary and x2 > right_region_boundary:
                        right_fit.append((slope, intercept))

        left_fit_average = np.average(left_fit, axis=0)
        if len(left_fit) > 0:
            x1 = (min_y - left_fit_average[1]) / left_fit_average[0]
            x2 = (max_y - left_fit_average[1]) / left_fit_average[0]
            cords.append([[int(x1), int(min_y), int(x2), int(max_y)]])
            ml = 1
        else:
            ml = 0

        right_fit_average = np.average(right_fit, axis=0)
        if len(right_fit) > 0:
            x1 = (min_y - right_fit_average[1]) / right_fit_average[0]
            x2 = (max_y - right_fit_average[1]) / right_fit_average[0]
            cords.append([[int(x1), int(min_y), int(x2), int(max_y)]])
            mr = 1
        else:
            mr = 0

        # print(ml, mr)
        return cords, ml, mr
    except:
        return 0, 0, 0

def draw_lines(img, lines):
    try:
        for line in lines:
            points = line[0]
            cv2.line(img, (points[0], points[1]), (points[2], points[3]), [0, 255, 0], 3)
    except:
        pass

def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def goStraight(m12):
    if m12 > 0.4:
        keyboard.press('W')
        time.sleep((m12 * 1/3))
        keyboard.release('W')
        keyboard.release('d')
        keyboard.release('A')
        print([0, 1, 0])


def goLeft(m1, m2):
    if m1 > 0.12:
        # keyboard.release('W')
        keyboard.press('A')
        time.sleep(m1*1.35)
        keyboard.release('A')
        keyboard.release('d')
        print([1, 0, 0])


def goRight(m1, m2):
    if m2 > 0.1:
        # keyboard.release('W')
        keyboard.press('d')
        time.sleep(m2*1.35)
        keyboard.release('d')
        keyboard.release('A')
        print([0, 0, 1])

def analize_pic():
    img = np.array(sct.grab(monitor))
    img_resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (48, 64))
    img_resized = np.expand_dims(img_resized, axis=0)

    return img_resized,img


model = tf.keras.models.load_model(r"C:\Users\Mondi\PycharmProjects\AI_Car\mymodel11.h5")

with mss.mss() as sct:
    # Part of the screen to capture
    # monitor = {"top": 30, "left": -1920, "width": 640, "height": 480}
    monitor = {"top": 30, "left": 0, "width": 640, "height": 480}

    while "Screen capturing":
        last_time = time.time()
        threadLock = threading.Lock()
        threads = []

        img_resized,img = analize_pic()
        ypred = model.predict(img_resized)[0]
        print(ypred)
        # Create new threads
        thread2 = myThreadLeft(ypred[0], ypred[2])
        thread3 = myThreadRight(ypred[0], ypred[2])
        thread1 = myThreadStraight(ypred[1])

        # Start new Threads
        thread1.start()
        thread2.start()
        thread3.start()

        # Add threads to thread list
        threads.append(thread1)
        threads.append(thread2)
        threads.append(thread3)

        # Wait for all threads to complete
        for t in threads:
            t.join()
        # Display the picture in HSV
        # cv2.imshow('OpenCV/Numpy grayscale1', mask)
        # cv2.imshow('OpenCV/Numpy grayscale2', processed_img)
        cv2.imshow('OpenCV/Numpy grayscale3', img)
        # cv2.imshow('OpenCV/Numpy grayscale4', cropped_edges2)

        # print("time: {}".format(time.time() - last_time))
        print("fps: {}".format(1 / (time.time() - last_time)))

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
