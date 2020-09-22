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
from lcmtypes import mbot_motor_pwm_t
from lcmtypes import simple_motor_command_t

# def row_check(bi_img, potential_col):
#     # input:
#     # bi_img: binary image
#     # potential_col: list of np array([row, col]), 0 to n-1 (from smaller row to larger row)
    
#     # output:
#     # np.array([leftmost col, rightmost col, center row])
#     start_col = potential_col[0][1]
#     start_row = potential_col[0][0]
#     end_row = potential_col[-1][0]
#     if (end_row-start_row > 100 or end_row-start_row < 3):
#         return np.array([-1, -1, -1])
#     row_len = end_row-start_row+1
#     cur_col = start_col
#     step = 3
#     while cur_col < bi_img.shape[1]-step and bi_img[start_row:end_row+1, cur_col:cur_col+1].sum() > row_len/4:
#         cur_col = cur_col+step
#     if cur_col - start_col > 3*(end_row-start_row):
#         return np.array([start_col, cur_col, int((start_row+end_row)/2)])
#     else:
#         return np.array([-1, -1, -1])
    
# def col_check(bi_img, potential_row):
#     # input:
#     # bi_img: binary image
#     # potential_row: list of np array([row, col]), 0 to n-1 (from smaller col to larger col)
    
#     # output:
#     # np.array([smallest row, largest row, center col])
#     start_row = potential_row[0][0]
#     start_col = potential_row[0][1]
#     end_col = potential_row[-1][1]
#     if (end_col-start_col > 100 or end_col-start_col < 3):
#         return np.array([-1, -1, -1])
#     col_len = end_col-start_col+1
#     cur_row = start_row
#     step = 3
#     while cur_row < bi_img.shape[0]-step and bi_img[cur_row:cur_row+1, start_col:end_col+1].sum() > col_len/4:
#         cur_row = cur_row+step
#     if cur_row - start_row > 3*(end_col-start_col):
#         return np.array([start_row, cur_row, int((start_col+end_col)/2)])
#     else:
#         return np.array([-1, -1, -1])

def find_potential_row_by_horizontal_line(bi_img, hori):
    row_to_check = max(hori[2] - 20, 0);
    start_col = hori[0]
    end_col = hori[1]
    potential_row = -1*np.ones((2, 2))
    step = 3
    start = False
    while potential_row[0, 0] == -1:
        if (row_to_check > hori[2]-10 or row_to_check >= bi_img.shape[0] or row_to_check < 0):
            break
        for col in range(start_col, end_col):
            if bi_img[row_to_check, col] > 0.5 and not start:
                start = True
                potential_row[0, 0] = row_to_check
                potential_row[0, 1] = col
            elif bi_img[row_to_check, col] < 0.5 and start:
                potential_row[1, 0] = row_to_check
                potential_row[1, 1] = col
                break
        row_to_check = row_to_check+step
    return potential_row

# def find_potential_cols_at_one_col(bi_img, col):
#     cols_list = []
#     temp_list = []
#     start = False
#     for row in range(bi_img.shape[0]):
#         if bi_img[row, col] > 0.5 and not start:
#             start = True
#             temp_list.append(np.array([row, col]))
#         elif bi_img[row, col] > 0.5 and start:
#             temp_list.append(np.array([row, col]))
#         elif bi_img[row, col] < 0.5 and start:
#             start = False
#             cols_list.append(temp_list)
#             temp_list = []
#     return cols_list

def remove_cross(bi_img, hori, vert):
    for row in range(vert[0], vert[1]):
        for col in range(hori[0], hori[1]):
            bi_img[row, col] = 0
    return bi_img

def draw_centers(img, centers):
    for center in centers:
        # print(center)
        for row in range(center[0]-2, center[0]+2):
            for col in range(center[1]-2, center[1]+2):
                img[row, col] = (0, 0, 255)
                # print(center)
    return img

def draw_line(img, trajectories):
    # pt: np.array([row, col])
    for pair in trajectories:
        pt1 = pair[0]
        pt2 = pair[1]
        for t in range(51):
            ratio = float(t)/50.0
            pt = pt1*ratio + pt2*(1-ratio)
            # print(pt)
            img[int(pt[0]), int(pt[1])] = (255, 0, 0)
    return img

def search_full_image(bi_img):
    box = np.zeros((2, 2))
    box[0, 0] = 0
    box[0, 1] = 0
    box[1, 0] = bi_img.shape[0]
    box[1, 1] = bi_img.shape[1]
    return search_partial_image(bi_img, box)
    # step = 3
    # np.savetxt("binary.csv", bi_img.astype(int), fmt='%d')
    # centers = []
    # for col in range(0, bi_img.shape[1], step):
    #     potential_cols = find_potential_cols_at_one_col(bi_img, col)
    #     for potential_col in potential_cols:
    #         hori = row_check(bi_img, potential_col)
    #         if (hori[0] > 0):
    #             potential_row = find_potential_row_by_horizontal_line(bi_img, hori)
    #             if potential_row:
    #                 vert = col_check(bi_img, potential_row)
    #                 if vert[0] > 0:
    #                     centers.append(np.array([hori[2], vert[2]]))
    #                     print("width of ----")
    #                     start_row = potential_col[0][0]
    #                     end_row = potential_col[-1][0]
    #                     print(end_row - start_row)
    #                     print("----:")
    #                     print(hori)

    #                     print("width of |")
    #                     start_col = potential_row[0][1]
    #                     end_col = potential_row[-1][1]
    #                     print(end_col - start_col)
    #                     print("|")
    #                     print(vert)
    #                     bi_img = remove_cross(bi_img, hori, vert)
    # return centers

def row_check_partial(bi_img, potential_col, box):
    # input:
    # bi_img: binary image
    # potential_col: 2-by-2 np array([row, col])
    # potential_col[0]: smaller row
    # potential_col[1]: larger row
    
    # output:
    # np.array([leftmost col, rightmost col, center row])
    start_col = int(potential_col[0][1])
    start_row = int(potential_col[0][0])
    end_row = int(potential_col[1][0])
    if (end_row-start_row > 100 or end_row-start_row < 3):
        return np.array([-1, -1, -1])
    row_len = end_row-start_row+1
    cur_col = start_col
    step = 3
    while cur_col < box[1, 1]-step and bi_img[start_row:end_row+1, cur_col:cur_col+1].sum() > row_len/4:
        cur_col = cur_col+step
    if cur_col - start_col > 3*(end_row-start_row):
        return np.array([start_col, cur_col, int((start_row+end_row)/2)])
    else:
        return np.array([-1, -1, -1])
    
def col_check_partial(bi_img, potential_row, box):
    # input:
    # bi_img: binary image
    # potential_row: 2-by-2 np array([row, col])
    # potential_row[0]: smaller col
    # potential_row[1]: larger col
    
    # output:
    # np.array([smallest row, largest row, center col])
    start_row = int(potential_row[0][0])
    start_col = int(potential_row[0][1])
    end_col = int(potential_row[1][1])
    if (end_col-start_col > 100 or end_col-start_col < 3):
        return np.array([-1, -1, -1])
    col_len = end_col-start_col+1
    cur_row = start_row
    step = 3
    while cur_row < box[1, 0]-step and bi_img[cur_row:cur_row+1, start_col:end_col+1].sum() > col_len/4:
        cur_row = cur_row+step
    if cur_row - start_row > 3*(end_col-start_col):
        return np.array([start_row, cur_row, int((start_col+end_col)/2)])
    else:
        return np.array([-1, -1, -1])

def find_potential_cols_at_one_col_partial(bi_img, col, box):
    cols_list = []
    one_col = np.zeros((2, 2))
    start = False
    for row in range(int(box[0, 0]), int(box[1, 0])):
        if bi_img[row, col] > 0.5 and not start:
            start = True
            one_col[0, 0] = row
            one_col[0, 1] = col
        elif bi_img[row, col] < 0.5 and start:
            one_col[1, 0] = row
            one_col[1, 1] = col
            start = False
            cols_list.append(np.copy(one_col))
    return cols_list

def search_partial_image(bi_img, box):
    # box: 2-by-2 array
    # [[least row, least col],
    #  [most row, most col]]
    step = 3
    centers = []
    for col in range(int(box[0, 1]), int(box[1, 1]), step):
        potential_cols = find_potential_cols_at_one_col_partial(bi_img, col, box)
        for potential_col in potential_cols:
            hori = row_check_partial(bi_img, potential_col, box)
            if (hori[0] > 0):
                potential_row = find_potential_row_by_horizontal_line(bi_img, hori)
                # if col == 246:
                #     print(potential_col)
                #     print(hori)
                #     print(potential_row)
                if potential_row[0, 0] is not -1:
                    vert = col_check_partial(bi_img, potential_row, box)
                    width_hori = potential_col[1][0] - potential_col[0][0]
                    width_vert = potential_row[1][1] - potential_row[0][1]
                    if vert[0] > 0 and vert[2] > hori[0] and vert[2] < hori[1] and hori[2] > vert[0] and hori[2] < vert[1] and width_hori < 3*width_vert and width_vert < 3*width_hori:
                        final_box = np.zeros((2, 2))
                        final_box[0 ,0] = max(0, hori[2]-2*width_hori)
                        final_box[0 ,1] = max(0, vert[2]-2*width_vert)
                        final_box[1, 0] = min(bi_img.shape[0], hori[2]+2*width_hori)
                        final_box[1, 1] = min(bi_img.shape[1], vert[2]+2*width_vert)
                        center = final_locate(bi_img, final_box.astype(int), width_hori, width_vert)
                        centers.append(center)
                        # print("width of ----")
                        # print(width_hori)
                        # print("----:")
                        # print(hori)

                        # print("width of |")
                        # print(width_vert)
                        # print("|")
                        # print(vert)
                        bi_img = remove_cross(bi_img, hori, vert)
    return centers

def construct_box(center, ROW, COL):
    bound_len = 100
    box = np.zeros((2, 2))
    box[0, 0] = max(center[0]-bound_len, 0)
    box[0, 1] = max(center[1]-bound_len, 0)
    box[1, 0] = min(center[0]+bound_len, ROW)
    box[1, 1] = min(center[1]+bound_len, COL)
    return box.astype(int)
    
def test():
    img = cv2.imread("test5.jpeg")
    lower_rgb = (0, 0, 0)
    upper_rgb = (100, 100, 100)
    bi_img = cv2.inRange(img, lower_rgb, upper_rgb)
    cv2.imwrite("binary.png", bi_img)
    bi_img = bi_img/255
    # import time
    # t1 = time.time()
    centers = search_full_image(bi_img)
    # t2 = time.time()
    # print(t2-t1)
    cv2.imwrite("result.png", draw_centers(img, centers))

def final_locate(bi_img, box, width_hori, width_vert):
    # box: 2-by-2 array
    # [[least row, least col],
    #  [most row, most col]]
    row_centers = []
    col_centers = []
    # if (box[0,0] == 233):
    #     np.savetxt("ginal_locate.txt", bi_img[box[0,0]:box[0,1], box[1,0]:box[1,1]], fmt = '%d')
    for row in range(box[0,0], box[1,0]):
        if bi_img[row:row+1, box[0,1]:box[1,1]].sum() > 0.8*(box[1,1]-box[0,1]):
            row_centers.append(row)
    for col in range(box[0,1], box[1,1]):
        if bi_img[box[0,0]:box[1,0], col:col+1].sum() > 0.8*(box[1,0]-box[0,0]):
            col_centers.append(col)
    if row_centers and col_centers and 2*width_hori > (col_centers[-1]-col_centers[0]) and 2*width_vert > (row_centers[-1]-row_centers[0]):
        return np.array([(row_centers[-1]+row_centers[0])/2, (col_centers[-1]+col_centers[0])/2])
    else:
        return np.array([-1, -1])

def main():
    FWD_PWM_CMD = 0.3
    TURN_PWM_CMD = 0.3
    flip_h = 0
    flip_v = 0

    video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (640, 480))

    trajectories = [] # list of 2-by-2 numpy arrays
    # [[pt1_row, pt1_col],
    #  [pt2_row, pt2_col]]
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")
    pygame.init()
    pygame.display.set_caption("MBot TeleOp")
    screen = pygame.display.set_mode([640,480])
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 24
    rawCapture = PiRGBArray(camera, size=(640, 480))
    time.sleep(0.5)
    idx = 0
    centers = []
    # last_centers = []
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

        lower_rgb = (0, 0, 0)
        upper_rgb = (100, 100, 100)
        bi_img = cv2.inRange(image, lower_rgb, upper_rgb)
        bi_img = bi_img/255
        # for center in centers:
        #     last_centers.append(center)
        if idx % (camera.framerate*3) == 0:
            centers = search_full_image(bi_img)
            # for center in centers:
            #     for last_center in centers:
            #         if (last_center[0] - center[0] > -50 and last_center[0] - center[0] < 50 and last_center[1] - center[1] > -50 and last_center[1] - center[1] < 50):
            #             trajectories.append(np.concatenate((center.reshape((1,2)), last_center.reshape((1,2))), axis=0))
        elif idx % 3 == 0:
            new_centers = []
            for center in centers:
                box = construct_box(center, bi_img.shape[0], bi_img.shape[1])
                new_center = search_partial_image(bi_img, box)
                if new_center:
                    new_centers.append(new_center[0])
                #     trajectories.append(np.concatenate((center.reshape((1,2)), new_center[0].reshape((1,2))), axis=0))
            centers = []
            for new_center in new_centers:
                centers.append(new_center)
        image = draw_centers(image, centers)
        # print(trajectories)
        # image = draw_line(image, trajectories)
        video.write(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))

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
                
        # command = mbot_motor_pwm_t()
        # command.left_motor_pwm =  fwd * FWD_PWM_CMD - turn * TURN_PWM_CMD
        # command.right_motor_pwm = fwd * FWD_PWM_CMD + turn * TURN_PWM_CMD
        # lc.publish("MBOT_MOTOR_PWM",command.encode())
        command = simple_motor_command_t()
        # command.left_motor_pwm =  fwd * FWD_PWM_CMD - turn * TURN_PWM_CMD
        # command.right_motor_pwm = fwd * FWD_PWM_CMD + turn * TURN_PWM_CMD
        command.forward_velocity = fwd
        command.angular_velocity = turn
        lc.publish("MBOT_MOTOR_COMMAND_SIMPLE",command.encode())
        rawCapture.truncate(0)
        print("frame: ", idx)
        # print("centers: ", centers)
        idx = idx+1
    video.release()

if __name__ == "__main__":
    test()