import numpy as np
import cv2
import matplotlib.pyplot as plt

import glob

# Input: point_1 [point_1_y, point_1_x] and point_2 [point_2_y, point_2_x] (representing the end points of a contour)
# Output: point_1 and point_2 sorted in a consistent manner based on the extent point [extend_point_y, extend_point_x]
# It's ok if the input is [x,y] and not [y,x] !!! the result will still be consistent
def sort_end_points(point_1, point_2, extend_point):

    middle_point_y = (point_1[0] + point_2[0]) / 2
    middle_point_x = (point_1[1] + point_2[1]) / 2

    extend_point_y = extend_point[0]
    extend_point_x = extend_point[1]

    v_extend = [extend_point_y - middle_point_y, extend_point_x - middle_point_x]
    v_1 = [point_1[0] - middle_point_y, point_1[1] - middle_point_x]
    v_2 = [point_2[0] - middle_point_y, point_2[1] - middle_point_x]

    if np.cross(v_extend, v_1) < 0 and np.cross(v_extend, v_2) > 0:
        # Switch points
        tmp = point_1
        point_1 = point_2
        point_2 = tmp

    assert(not(np.cross(v_extend, v_1) < 0 and np.cross(v_extend, v_2) < 0) and not(np.cross(v_extend, v_1) > 0 and np.cross(v_extend, v_2) > 0))

    return point_1, point_2


# Input: a point and a contour
# Output: the point in the contour that is closest to the input point
def find_closest_in_contour(point, contour):

    assert(contour.shape[0] > 0)

    min_dist = 1000000
    min_point = None
    for i in range(contour.shape[0]):
        dist = pow(pow(point[0] - contour[i][0], 2) + pow(point[1] - contour[i][1], 2), 0.5)
        if dist < min_dist:
            min_dist = dist
            min_point = contour[i]
    point = min_point

    return point

# Input: 'inner'/'outer' string, three points (two end points and one central opening point), and a contour
# Output: the inner or outer(opening) points of contour
# contour is assumed to be an "open/closed" contour with 3 opening points
def get_contour_segment(segment_type, point_1, point_2, point_3, contour):

    assert(contour.shape[0] > 0)

    # Sometimes in sax biobank we have point_1 == point_2 == point_3
    # which lead to index_1 == index_2 == index_3

    index_1 = -1
    index_2 = -1
    index_3 = -1
    for i in range(contour.shape[0]):
        if contour[i][0] == point_1[0] and contour[i][1] == point_1[1]:
            index_1 = i
        if contour[i][0] == point_2[0] and contour[i][1] == point_2[1]:
            index_2 = i
        if contour[i][0] == point_3[0] and contour[i][1] == point_3[1]:
            index_3 = i
        if index_1 != -1 and index_2 != -1 and index_3 != -1:
            break

    assert(index_1 != -1 and index_2 != -1 and index_3 != -1)

    # making sure index_1 <= index_2
    if index_1 > index_2:
        tmp = index_1
        index_1 = index_2
        index_2 = tmp

    assert(segment_type == 'inner' or segment_type == 'outer')

    contour_segment = []

    if (index_2 - index_1 + 1) < (0.2 * contour.shape[0]):
        if segment_type == 'outer':
            for i in range(index_1, index_2 + 1):
                contour_segment.append(contour[i])
        elif segment_type == 'inner':
            for i in range(index_2, contour.shape[0]):
                contour_segment.append(contour[i])
            for i in range(0, index_1 + 1):
                contour_segment.append(contour[i])
    elif (index_2 - index_1 + 1) > (0.8 * contour.shape[0]):
        if segment_type == 'outer':
            for i in range(index_2, contour.shape[0]):
                contour_segment.append(contour[i])
            for i in range(0, index_1 + 1):
                contour_segment.append(contour[i])
        elif segment_type == 'inner':
            for i in range(index_1, index_2 + 1):
                contour_segment.append(contour[i])
    else:
        # the two segments are of similar length
        # will use the third point to determine the inner/outer segment
        if index_1 <= index_3 <= index_2:
            if segment_type == 'outer':
                for i in range(index_1, index_2 + 1):
                    contour_segment.append(contour[i])
            elif segment_type == 'inner':
                for i in range(index_2, contour.shape[0]):
                    contour_segment.append(contour[i])
                for i in range(0, index_1 + 1):
                    contour_segment.append(contour[i])
        else:
            if segment_type == 'outer':
                for i in range(index_2, contour.shape[0]):
                    contour_segment.append(contour[i])
                for i in range(0, index_1 + 1):
                    contour_segment.append(contour[i])
            elif segment_type == 'inner':
                for i in range(index_1, index_2 + 1):
                    contour_segment.append(contour[i])

    return contour_segment

# Inserts a mask of contour into seg using label
# contour is expected to be closed
# Returns seg and the "fixed" contour (after correcting for cvi coordinate issue)
def insert_closed_contour_to_mask(contour, seg, label):

    seg_tmp = np.zeros(shape=seg.shape, dtype = np.uint8)
    for i in range(len(contour)):
        # Moving from xml to numpy space
        seg_tmp[contour[i][1], contour[i][0]] = 1

    #plt.imshow(seg_tmp, cmap='gray')
    #plt.show()

    # Filling the area of the current closed contour (lv_open is also closed)
    _, contours, _ = cv2.findContours(seg_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert(len(contours) == 1)
    cv2.drawContours(seg_tmp, contours, 0, 1, cv2.FILLED);

    #plt.imshow(seg_tmp, cmap='gray')
    #plt.show()

    # Fixing the coordinate system problem
    seg_tmp = fix_coordinate_issue(seg_tmp)

    #plt.imshow(seg_tmp, cmap='gray')
    #plt.show()

    # fix_coordinate_issue() can cause the segmentation to disapeer when it's very small
    if np.count_nonzero(seg_tmp) == 0:
        return seg, None

    # fix_coordinate_issue() might create single pixels at the boundary of the big mask
    # these single pixels will be added to the segmentation but ignored from the contour
    _, contours, _ = cv2.findContours(seg_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_size = -1
    max_index = -1
    for i in range(len(contours)):
        if len(contours[i]) > max_size:
            max_size = len(contours[i])
            max_index = i

    fixed_contour = []
    for i in range(max_size):
        fixed_contour.append(contours[max_index][i][0])

    seg_tmp = seg_tmp * label

    # seg_available is all available pixels in seg
    seg_available = np.copy(seg)
    seg_available = 1 - np.clip(seg_available, 0, 1)

    seg = seg + (seg_tmp * seg_available)

    return seg, np.array(fixed_contour)


# Closing, filling, and inserting contour into seg using helper_contour and its 3 opening points
# Example for this function is (contour == lv_epi_open_points) and (helper_contour == new_lv_endo)
def insert_open_contour_to_mask(p1, p2, contour, h_p1, h_p2, h_p3, helper_contour, seg, label):

    seg_tmp = np.zeros(shape=seg.shape, dtype = np.uint8)

    # contour (original points)
    for i in range(len(contour)):
        # Moving from xml to numpy space
        seg_tmp[contour[i][1], contour[i][0]] = 1

    #print('SEG')
    #plt.imshow(seg, cmap='gray')
    #plt.show()

    #print('just epi')
    #plt.imshow(seg_tmp, cmap='gray')
    #plt.show()

    # Getting the inner/long helper contour, closest_p1, and closest_p2
    inner_helper_contour = get_contour_segment('inner', h_p1, h_p2, h_p3, helper_contour)
    closest_p1 = find_closest_in_contour(p1, np.array(inner_helper_contour))
    closest_p2 = find_closest_in_contour(p2, np.array(inner_helper_contour))

    # Getting and drawing the opening/short helper contour
    outer_helper_contour = get_contour_segment('outer', closest_p1, closest_p2, h_p3, helper_contour)
    for i in range(len(outer_helper_contour)):
        # Moving from xml to numpy space
        seg_tmp[outer_helper_contour[i][1], outer_helper_contour[i][0]] = 1

    #print('epi+endo_opening')
    #plt.imshow(seg_tmp, cmap='gray')
    #plt.show()

    # Connecting between closest_p1 and p1 AND closest_p2 and p2
    for pair in [[closest_p1, p1],[closest_p2, p2]]:
        seg_tmp = insert_line_to_mask(pair, seg_tmp)

    #print('closed contour')
    #plt.imshow(seg_tmp, cmap='gray')
    #plt.show()

    # Filling the area of the current closed contour
    _, contours, _ = cv2.findContours(seg_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert(len(contours) == 1)
    cv2.drawContours(seg_tmp, contours, 0, 1, cv2.FILLED);

    #print('filled')
    #plt.imshow(seg_tmp, cmap='gray')
    #plt.show()

    # Fixing the coordinate system problem
    seg_tmp = fix_coordinate_issue(seg_tmp)

    #print('filled fixed')
    #plt.imshow(seg_tmp, cmap='gray')
    #plt.show()

    seg_tmp = seg_tmp * label

    # seg_available is all available pixels in seg
    seg_available = np.copy(seg.astype(np.uint8))
    seg_available = 1 - np.clip(seg_available, 0, 1)

    seg_tmp = seg_tmp * seg_available

    # seg_tmp is now myo only

    #plt.imshow(seg_tmp, cmap='gray')
    #plt.show()

    # Getting fixed_contour from the myo mask only (without the endo)
    # Also, fix_coordinate_issue() might break the myo mask into multiple components
    # So we take the boundary points of all contours to make sure the opening landmark points are adjusted properly
    _, contours, _ = cv2.findContours(seg_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fixed_contour = []
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            fixed_contour.append(contours[i][j][0])

    seg = seg + seg_tmp # seg_tmp has already went through intersection with seg_available

    return seg, np.array(fixed_contour)

# Inserts a line to seg between the two points in pair
def insert_line_to_mask(pair, seg):
    y_1 = pair[0][1]
    y_2 = pair[1][1]
    x_1 = pair[0][0]
    x_2 = pair[1][0]
    if ((x_1 - x_2) != 0):
        m = (y_1 - y_2) / (x_1 - x_2)
        b = y_1 - (m * x_1)
        for x in range(int(min(x_1, x_2)), int(max(x_1 + 1, x_2 + 1))):
            y = int(round((m * float(x)) + b))
            seg[y,x] = 1 # Moving from xml to numpy space
        for y in range(int(min(y_1, y_2)), int(max(y_1 + 1, y_2 + 1))):
            if m != 0:
                x = int(round((float(y) - b) / m))
                seg[y,x] = 1 # Moving from xml to numpy space
    else:
        for y in range(int(min(y_1, y_2)), int(max(y_1 + 1, y_2 + 1))):
            seg[y, x_1] = 1 # Moving from xml to numpy space

    return seg


def fix_coordinate_issue(seg):
    rows = seg.shape[0]
    cols = seg.shape[1]
    w = cv2.warpAffine(seg, np.float32([[1,0,-1],[0,1,0]]), (cols,rows))
    n = cv2.warpAffine(seg, np.float32([[1,0,0],[0,1,-1]]), (cols,rows))
    nw = cv2.warpAffine(seg, np.float32([[1,0,-1],[0,1,-1]]), (cols,rows))
    seg = (seg * w * n * nw)

    return seg


# Returning the largest contour in mask with the value 'label'
# mask is 2D
def create_two_contours_from_mask(mask, label):
    # need to fix the coordinate issue by adding the southeast pixels
    binary_mask = (mask == label)
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_size1 = -1
    max_cont1 = None
    max_size2 = -1
    max_cont2 = None
    for i in range(len(contours)):
        if len(contours[i]) > max_size1:
            if max_size1 > max_size2:
                max_size2 = max_size1
                max_cont2 = max_cont1
            max_size1 = len(contours[i])
            max_cont1 = contours[i]
        elif len(contours[i]) > max_size2:
            max_size2 = len(contours[i])
            max_cont2 = contours[i]

    return max_cont1, max_cont2
