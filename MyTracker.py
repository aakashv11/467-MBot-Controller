import numpy as np
import matplotlib.pyplot as plt

general_search_offset = 3 # full-image scan
local_search_offset = 1 # local scan
larger_bound_offset = 5 # when a point meets our requirement, we want to search in a larger area
                        # Prevent missing features during general_search
threshold = 45 # determine the boundary
white_min_value = 110 # determine the blackpixels are end near the boundary

class Tracker:
    def __init__(self):
        # each target has the following structure:
        # [row, column, V_top, V_bot, H_left, H_right]
        self.targets = []

    def search(self, image_grey, row_box, column_box):

        for row in range(row_box[0] + general_search_offset, row_box[1] - general_search_offset, general_search_offset):
            for column in range(column_box[0] + general_search_offset, column_box[1] - general_search_offset, general_search_offset):

                if column <= 0 or column >=480 or row >= 640 or row <= 0: continue

                # If the target pixel falls in any targets' area, then we directly stop searching
                stop_search = False
                for item in self.targets:
                    if row < item[3] and row > item[2] and column > item[4] and column < item[5]:
                        stop_search = True
                if stop_search: continue

                # Parameters for Vertical Component
                # V_top is the row of the top of the vertical Component
                # V_bot is the row of the bottom of the vertical Component
                # V_left and V_right are lists which contain all detected boundary points
                # The number of points in the list are set to be equal by using flags in the code
                V_top, V_bot = [None, None]
                V_left = []
                V_right = []
                V_center = [None, None]

                gradient = image_grey[row, column - general_search_offset] - image_grey[row, column]
                if gradient > threshold:
                    # Large change here, vertical boundry detected
                    # We need a local search to determine whether this is a place where we can possibly detect Vertical_component

                    # Find the left end of the V_top
                    current_left_end = None
                    for columnindex in range(column - larger_bound_offset, column + larger_bound_offset, local_search_offset):

                        if columnindex >= column_box[1] or columnindex < column_box[0]: continue
                        if columnindex <= 0 or columnindex >=480 : continue

                        gradient = image_grey[row, columnindex - local_search_offset] - image_grey[row, columnindex]
                        if gradient > threshold and current_left_end is None:
                            current_left_end = columnindex
                            break;

                    # Find the right end of the V_top
                    current_right_end = None
                    for local_column in range(column, column_box[1] - local_search_offset, local_search_offset):

                        if local_column >= column_box[1] or local_column < column_box[0] or local_column >= 480: continue
                        #if local_column <= 0 or local_column >=480 : continue

                        other_gradient = image_grey[row, local_column] - image_grey[row, local_column - local_search_offset]

                        if other_gradient > threshold:
                            current_right_end = local_column
                            break;

                    if current_right_end is None or current_left_end is None:
                        # The candidate is possibly not a V_component
                        continue

                    # Now we find a horizontal blackline. We need to search the nearby area to find a possible vertical component
                    for new_row in range(row - larger_bound_offset, row_box[1] - local_search_offset, local_search_offset):
                        vertical_edge1_flag = False
                        vertical_edge2_flag = False
                        row_min = 255
                        for new_column in range(current_left_end - larger_bound_offset, current_right_end + larger_bound_offset, local_search_offset):

                            if new_column >= column_box[1] - local_search_offset or new_column < column_box[0] + local_search_offset or new_row >= row_box[1] - local_search_offset or new_row <= row_box[0] + local_search_offset: continue
                            #if new_column <= 0 or new_column >=480 or new_row >= 640 or new_row <= 0: continue

                            if image_grey[new_row, new_column] - image_grey[new_row, new_column + local_search_offset] > threshold:

                                if not vertical_edge1_flag:
                                    vertical_edge1_flag = True
                                    V_left.append(new_column)
                                    # Here we need to update the current_left_end && current_right_end since the component may not be exactly vertical
                                    current_left_end = new_column

                                    if V_top is None:
                                        V_top = new_row

                            if image_grey[new_row, new_column  + local_search_offset] - image_grey[new_row, new_column] > threshold:
                                if vertical_edge1_flag and vertical_edge2_flag is False:
                                    vertical_edge2_flag = True
                                    V_right.append(new_column)
                                    # Here we need to update the current_left_end && current_right_end since the component may not be exactly vertical
                                    current_right_end = new_column

                            if row_min > image_grey[new_row, new_column]:
                                row_min = image_grey[new_row, new_column]


                        if row_min > white_min_value and V_top and V_bot is None:
                            V_bot = new_row
                            break;
                            # Finish detecting the vertical component

                    if V_top is None or V_bot is None or not V_left or not V_right or (V_bot - V_top < 30):
                        # Not found
                        continue;

                    V_left = np.array(V_left)
                    V_right = np.array(V_right)
                    V_center = [int((V_top + V_bot) // 2), int((V_left.mean() + V_right.mean()) // 2)]

                    #############################################
                    # Finish Vertical Component Detection
                    #############################################

                    # Parameters for Horizontal Component
                    # H_left is the column of the left of the horizontal Component
                    # H_right is the column of the right of the horizontal Component
                    # H_top and H_bot are lists which contain all detected boundary points
                    # The number of points in the list are set to be equal by using flags in the code
                    H_left, H_right = [None, None]
                    H_top = []
                    H_bot = []

                    column_width = int((V_right.mean() - V_left.mean()) // 2 )
                    current_top_end = V_center[0] - 2 * column_width
                    current_bot_end = V_center[0] + 2 * column_width

                    # Similar as before: search from the midline to the left to find the left end of the horizontal component
                    for new_column in range(V_center[1], column_box[0] + local_search_offset, -local_search_offset):
                        horizontal_edge1_flag = False
                        horizontal_edge2_flag = False
                        column_min = 255
                        for new_row in range(current_top_end - larger_bound_offset, current_bot_end + larger_bound_offset, local_search_offset):
                            if new_column >= column_box[1] - local_search_offset or new_column < column_box[0] + local_search_offset or new_row >= row_box[1] - local_search_offset or new_row <= row_box[0] + local_search_offset: continue

                            if image_grey[new_row - local_search_offset, new_column] - image_grey[new_row, new_column] > threshold:
                                if not horizontal_edge1_flag:
                                    horizontal_edge1_flag = True
                                    H_top.append(new_row)
                                    current_top_end = new_row

                            if image_grey[new_row, new_column] - image_grey[new_row - local_search_offset, new_column] > threshold:
                                if horizontal_edge1_flag and not horizontal_edge2_flag:
                                    horizontal_edge2_flag = True
                                    H_bot.append(new_row)
                                    current_top_end = new_row

                            if column_min > image_grey[new_row, new_column]:
                                column_min = image_grey[new_row, new_column]

                        if column_min > white_min_value and H_left is None:
                            H_left = new_column
                            break;
                            # finish detecting the horizontal component left end

                    if H_left is None: continue

                    column_width = int((V_right.mean() - V_left.mean()) // 2)
                    current_top_end = V_center[0] - 2 * column_width
                    current_bot_end = V_center[0] + 2 * column_width

                    # Similar as before: search from the midline to the right to find the left end of the horizontal component
                    for new_column in range(V_center[1], column_box[1], local_search_offset):

                        horizontal_edge1_flag = False
                        horizontal_edge2_flag = False
                        column_min = 255

                        for new_row in range(current_top_end - larger_bound_offset, current_bot_end + larger_bound_offset, local_search_offset):

                            if new_column >= column_box[1] - local_search_offset or new_column < column_box[0] + local_search_offset or new_row >= row_box[1] - local_search_offset or new_row <= row_box[0] + local_search_offset: continue
                            if image_grey[new_row - local_search_offset, new_column] - image_grey[new_row, new_column] > threshold:
                                if not horizontal_edge1_flag:
                                    horizontal_edge1_flag = True
                                    H_top.append(new_row)

                                    current_top_end = new_row

                            if image_grey[new_row, new_column] - image_grey[new_row - local_search_offset, new_column] > threshold:
                                if horizontal_edge1_flag and not horizontal_edge2_flag:
                                    horizontal_edge2_flag = True
                                    H_bot.append(new_row)

                                    current_bot_end = new_row

                            if column_min > image_grey[new_row, new_column]:
                                column_min = image_grey[new_row, new_column]

                        if column_min > white_min_value and H_right is None:
                                H_right = new_column
                                break;
                                # finish detecting the horizontal component right end


                    if H_left is None or H_right is None or not H_bot or not H_top:
                        # Not found
                        continue;

                    if image_grey[V_bot, H_left] < white_min_value or image_grey[V_bot, H_right] < white_min_value or image_grey[V_top, H_left] < white_min_value or image_grey[V_top, H_right] < white_min_value:
                        continue;

                    #print("Find a target!")
                    self.targets.append([int((V_top + V_bot)//2), int((H_left + H_right)//2), V_top, V_bot, H_left, H_right])

                    #############################################
                    # Finish Target Detection
                    #############################################


    def draw_centers(self, image):
        for target in self.targets:
            for row in range(target[0] - 3, target[0] + 3):
                for column in range(target[1] - 3, target[1] + 3):
                    image[row, column] = (255, 0, 0)


    def draw_others(self, image):
        # draw the bounding boxes for each target
        for target in self.targets:
                image[target[2],:] = (255, 0, 0)
                image[target[3],:] = (255, 0, 0)
                image[:,target[4]] = (0, 255, 0)
                image[:,target[5]] = (0, 255, 0)


    def get_targets(self):
        return self.targets


    def empty_targets(self):
        self.targets = []