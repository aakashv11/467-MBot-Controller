import pygame
from pygame.locals import *
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import numpy as np
import sys
import math

sys.path.append("lcmtypes")
import lcm
from lcmtypes import simple_motor_command_t
FWD_PWM_CMD = 0.3
TURN_PWM_CMD = 0.3
NUM_ROWS = 640
NUM_COLS = 480
BLACK_LIM = 45
WHITE_LIM = 100

def check_center_color(im, center_x, center_y):
     # Make sure center is black
    if not BLACK_LIM > im[center_y][center_x]:
        return False
    # Make sure up, left, down , right are black
    if not BLACK_LIM > im[center_y-2][center_x]:
        return False
    if not BLACK_LIM > im[center_y+2][center_x]:
        return False
    if not BLACK_LIM > im[center_y][center_x-2]:
        return False
    if not BLACK_LIM > im[center_y][center_x+2]:
        return False


    #Check contrast:
    # Check up and left
    if not (WHITE_LIM < im[center_y-5][center_x-5]):
        return False
    # check up and right
    if not (WHITE_LIM < im[center_y+5][center_x-5]):
        return False
    # Check down and left
    if not (WHITE_LIM < im[center_y-5][center_x+5]):
        return False
    #Check down and right
    if not (WHITE_LIM < im[center_y+5][center_x+5]):
        return False

    return True

def line_intersect(m1, b1, m2, b2):
    if m1 == m2:
        return None
    x = (b2-b1)/(m1-m2)
    
    y = m1*x+b1
    return x,y

def best_fit(xs,ys):
    mean_xs = np.mean(xs)
    mean_ys = np.mean(ys)
    m = (((mean_xs*mean_ys) - np.mean(xs*ys))/
         ((mean_xs*mean_xs) - np.mean(xs*xs)))
    b = mean_ys - m*mean_xs
    return m, b

# Add all black points on horizontal line to list and return
def find_horizontal_line(im, row_lims, col_lims):
    pts = []
    for row in range(row_lims[0], row_lims[1], 2):
        for col in range(col_lims[0], col_lims[1], 1):
            if BLACK_LIM > im[row][col]:
                pts.append((col, row))
    return pts

# Add all black points on vertical line to list and return
def find_vertical_line(im, row_lims, col_lims):
    # Start row and col for searching for vertical line
    start_row, end_row = row_lims[0], row_lims[1]
    start_col, end_col = col_lims[0], col_lims[1]
    pts = []
    if start_col >= NUM_COLS or end_col >= NUM_COLS:
        return pts
    
    # Should begin search in white
    if not WHITE_LIM < im[start_row][start_col]:
        return pts

    vert_row = -1
    vert_col = -1

    # Find candidate row and col to start search for vertical line
    for row in range(start_row, end_row, 1):
        if BLACK_LIM > im[row][start_col]:
            vert_row = row
            vert_col = start_col
            break

    #No close to black pixel found
    if vert_row == -1:
        return pts
    
    if vert_row + 9 >= NUM_ROWS:
        return pts

    if vert_col - 25 < 0:
        vert_col = 25
    if vert_col + 30 >= NUM_COLS:
        vert_col = NUM_COLS-30

    #Find points for vertical line
    for col in range(vert_col - 20, vert_col + 30, 2):
        for row in range(vert_row-3, vert_row+5, 1):
            if BLACK_LIM > im[row][col]:
                pts.append((col, row))
    return pts

# Rows in im matrix are 1st dim, col in im matrix are second dim
# Axes are swapped in image so col acts as X, row acts as Y
# Axes swapped in pygame camera gui so rows are on horizontal (y axis)
# cols are on vertical (x axis)
def check_candidate(im, start_row, start_col):
    vertical_line_pts = []
    horizontal_line_pts = []
    rows,cols,channels = im.shape

    invalid = [-1,-1]

    #too far right to work
    if start_row + 40 >= rows or start_row - 20 < 0:
        return invalid

    # Too high or low
    if start_col - 5 < 0 or start_col + 5 >= cols:
        return invalid
    
    # Make sure pixels above start are not too dark, look for contrast
    if not WHITE_LIM < im[start_row][start_col - 5]:
        return invalid
    
    row_lims = [start_row - 20, start_row + 40]

    col_lims = [start_col - 3, start_col+4]
    # Find horizontal line
    horizontal_line_pts = find_horizontal_line(im, row_lims, col_lims)
    
    #Check where there are sufficient points
    if(len(horizontal_line_pts) < 3):
        return invalid
    
    # Check whether it is at least 20 pixels long
    if(horizontal_line_pts[-1][1] - horizontal_line_pts[0][1] < 20):
        return invalid
    else:
        cv2.line(im, horizontal_line_pts[0], horizontal_line_pts[-1], (0, 255, 0), 2)
    

    row_lims = [horizontal_line_pts[0][1], horizontal_line_pts[-1][1]]
    col_lims = [horizontal_line_pts[0][0] - 5, horizontal_line_pts[0][0] + 5]

    vertical_line_pts = find_vertical_line(im, row_lims, col_lims)
    
    # Check whether there are sufficient points
    if(len(vertical_line_pts) < 3):
        return invalid
    
    # Check whether it is of required length
    if(vertical_line_pts[-1][0] - vertical_line_pts[0][0] < 20):
        return invalid
    else:
        cv2.line(im, vertical_line_pts[0], vertical_line_pts[-1], (0, 0, 255), 2)
    
    #Get x,y coords on vertical line
    vert_xs = [coord[0] for coord in vertical_line_pts]
    vert_ys = [coord[1] for coord in vertical_line_pts]
    
    # get x,y, coords on horizontal line
    hor_xs = [coord[0] for coord in horizontal_line_pts]
    hor_ys = [coord[1] for coord in horizontal_line_pts]
    
    # Determine lines of best fit for vertical and horizontal lines
    v_m, v_b = best_fit(np.array(vert_xs), np.array(vert_ys))
    h_m, h_b = best_fit(np.array(hor_xs), np.array(hor_ys))
    
    # Parallel
    if h_m == v_m:
        return invalid
    
    #Find where vertical and horizontal lines intersect
    center_x, center_y = line_intersect(v_m, v_b, h_m, h_b)

    # No intersection
    if math.isnan(center_x) or math.isnan(center_y):
        return invalid
    
    #Float to int
    center_x = int(center_x)
    center_y = int(center_y)
    
    
    # Center is too close to edge
    if (center_x-5 < 0 or center_y-5 < 0) or (center_x+5 >= 480 or center_y+5 >= 640):
        return invalid
        
    
    # Check whether they coordinates are within the bounds of the cross
    if center_x < vertical_line_pts[0][0] or center_x > vertical_line_pts[-1][0]:
        return invalid
        
    if center_y < horizontal_line_pts[0][1] or center_y > horizontal_line_pts[-1][1]:
        return invalid
    
    
    if not check_center_color(im, center_x, center_y):
        return invalid

    return [center_x, center_y]


def search_full_image(image):
    rows, cols, channels = image.shape
    targets = []
    for row in range(0, rows-6, 5):
        for col in range(0, cols - 5, 4):
            # Candidate target
            if BLACK_LIM > image[row][col]:
                center = check_candidate(image, row, col)
                if center != [-1,-1]:
                    targets.append(center)
    return targets


def search_partial_image(image, row_lims, col_lims):
    start_row, end_row = row_lims[0], row_lims[1]
    start_col, end_col = col_lims[0], col_lims[1]
    targets = []
    for row in range(start_row, end_row, 3):
        for col in range(start_col, end_col, 3):
            # Candidate target
            if BLACK_LIM > image[row][col]:
                center = check_candidate(image, row, col)
                if center != [-1,-1]:
                    targets.append(center)
    return targets

def construct_search_frame(center):
    bound = 100
    col_lims = [max(center[0]-bound, 0), min(center[0]+bound, NUM_COLS)]
    row_lims = [max(center[1]-bound, 0), min(center[1]+bound, NUM_ROWS)]
    return row_lims, col_lims

def draw_centers(im, centers):
    for center in centers:
         cv2.circle(im, (center[0], center[1]), 3, (255,0,0))
    return


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
centers = []
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    screen.fill([0,0,0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.swapaxes(0,1)
    image = cv2.flip(image, -1)

    image_grey = image.mean(2)
    centers = search_full_image(image_grey)
    
    if counter % (2 * camera.framerate) == 0:
    	centers = []
    	time_before = time.time()
    	centers = search_full_image(image_grey)
    	print(time.time() - time_before)
    elif counter % 3 == 0:
        new_centers = []
        for center in centers:
            # Construct search frame
            row_lims, col_lims = construct_search_frame(center)
            # Search for new center in search frame
            new_center = search_partial_image(image, row_lims, col_lims)
            if len(new_center) > 0:
                new_centers.append(new_center[0])
            centers = []
            centers.extend(new_centers)
    draw_centers(image, centers)

    image = pygame.surfarray.make_surface(image)
    screen.blit(image, (0,0))
    pygame.display.update()
    fwd_velocity = 0.0
    ang_velocity = 0.0
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            pygame.quit()
            sys.exit()
            cv2.destroyAllWindows()
            exit
    key_input = pygame.key.get_pressed()  
    if key_input[pygame.K_LEFT]:
        ang_velocity += 1.0
    if key_input[pygame.K_UP]:
        fwd_velocity +=1.0
    if key_input[pygame.K_RIGHT]:
        ang_velocity -= 1.0
    if key_input[pygame.K_DOWN]:
        fwd_velocity -= 1.0
    counter = counter+1
    command = simple_motor_command_t.simple_motor_command_t()
    command.forward_velocity =  fwd_velocity 
    command.angular_velocity = ang_velocity
    lc.publish("MBOT_MOTOR_COMMAND_SIMPLE", command.encode())
    rawCapture.truncate(0)
