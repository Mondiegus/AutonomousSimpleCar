import numpy as np
import cv2
import time
import mss
import os
import pygame

# Initialize the joysticks.

pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
print(joystick.get_name())
joystick.init()
with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {"top": 30, "left": 0, "width": 640, "height": 480}
    output_data = [0.0, 0.0]
    filenameX = f"D:\\PngOutput\\training_dataX.npy"
    filenameY = f"D:\\PngOutput\\training_dataY.npy"
    if os.path.isfile(filenameX):
        print("file exist, loading prev data!")
        training_dataX = list(np.load(filenameX))
        training_dataY = list(np.load(filenameY))
    else:
        print("file not exist, creating new one!")
        training_dataX = []
        training_dataY = []

    while "Screen capturing":
        last_time = time.time()

        output_pic = f"D:\\PngOutput\\sct-{last_time}.png"

        # # Grab the data
        sct_img = sct.grab(monitor)

        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(sct.grab(monitor))
        img_resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (48, 64))
        training_dataX.append([img_resized])

        pygame.event.get()

        # left X - -1 is left, 1 is right -> Steering wheel; right Y - -1 is up, 1 is down -> Acc
        training_dataY.append([[joystick.get_axis(0), joystick.get_axis(3)]])

        if len(training_dataY) % 500 == 0:
            print(len(training_dataY))
            np.save(filenameX, training_dataX)
            np.save(filenameY, training_dataY)

        # print("time: {}".format(time.time() - last_time))
        print("fps: {}".format(1 / (time.time() - last_time)))

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
