import pygame
from pygame.locals import *
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import numpy as np
import sys
sys.path.append("lcmtypes")
import lcm
from simplelcm import simple_motor_command_t
import MyTracker

FWD_PWM_CMD = 0.3
TURN_PWM_CMD = 0.3
flip_h = 0
flip_v = 0
local_search_parameter = 50

Mytracker = MyTracker.Tracker()

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")
pygame.init()
pygame.display.set_caption("MBot TeleOp")
screen = pygame.display.set_mode([640,480])
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.5)

counter = 0

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    if (flip_h == 1 & flip_v == 0):
        image = cv2.flip(image, 1)
    elif (flip_h == 0 & flip_v == 1):
        image = cv2.flip(image, 0)
    elif (flip_h == 1 & flip_v == 1):
        image = cv2.flip(image, -1)

    screen.fill([0,0,0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.swapaxes(0,1)
    image = cv2.flip(image, -1)
    image_grey = image.mean(2)

    if counter % (2 * camera.framerate) == 0:
        start_time = time.time()
        Mytracker.empty_targets()
        Mytracker.search(image_grey, [0, image.shape[0]], [0, image.shape[1]])
        end_time = time.time()
    elif counter % 3 == 0:
        target_list = Mytracker.get_targets()
        Mytracker.empty_targets()
        for target in target_list:
            Mytracker.search(image_grey, [target[2] - local_search_parameter, target[3] + local_search_parameter], [target[4] - local_search_parameter, target[5] + local_search_parameter])

    Mytracker.draw_centers(image)

    image = pygame.surfarray.make_surface(image)
    screen.blit(image, (0,0))
    pygame.display.update()
    fwd = 0.0
    turn = 0.0
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            pygame.quit()
            sys.exit()
            cv2.destroyAllWindows()
    key_input = pygame.key.get_pressed()
    if key_input[pygame.K_LEFT]:
        turn += 1.0
    if key_input[pygame.K_UP]:
        fwd +=1.0
    if key_input[pygame.K_RIGHT]:
        turn -= 1.0
    if key_input[pygame.K_DOWN]:
        fwd -= 1.0
    if key_input[pygame.K_h]:
        if flip_h == 0:
            flip_h = 1
        else:
            flip_h = 0
    if key_input[pygame.K_v]:
        if flip_v == 0:
            flip_v = 1
        else:
            flip_v = 0
    if key_input[pygame.K_q]:
            pygame.quit()
            sys.exit()
            cv2.destroyAllWindows()

    #print("Frame:", counter)

    counter = counter + 1

    command = simple_motor_command_t()
    command.forward_velocity = fwd
    command.angular_velocity = turn
    lc.publish("MBOT_MOTOR_COMMAND_SIMPLE",command.encode())
    rawCapture.truncate(0)