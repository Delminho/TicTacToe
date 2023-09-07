import cv2
import numpy as np
import skimage
from skimage.morphology import skeletonize


def hough_inter(rho1, theta1, rho2, theta2):
    """
    Finds the intersection of 2 Hough Lines
    Returns np.array([x, y])
    """
    A = np.array([[np.cos(theta1), np.sin(theta1)], 
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([rho1, rho2])
    return np.linalg.lstsq(A, b)[0].astype(int)


def find_y_given_x(x, point1, point2):
    """
    Given x and two points of a line, find the y corresponding to x on this line
    """
    (x_1, y_1), (x_2, y_2) = point1, point2
    m = (y_2 - y_1) / (x_2 - x_1)
    b = y_1 - m * x_1
    # Use the equation of the line to find y_point
    y_point = m * x + b
    return int(y_point)

def find_x_given_y(y, point1, point2):
    """
    Given y and two points of a line, find the x corresponding to y on this line
    """
    (x_1, y_1), (x_2, y_2) = point1, point2
    m = (y_2 - y_1) / (x_2 - x_1 + 0.000001) # Avoid division by zero
    b = y_1 - m * x_1
    # Use the equation of the line to find y_point
    x_point = (y-b)/(m+0.0001) # Avoid division by zero
    return int(x_point)


def parallel_line(point, point1, point2):
  """
  This function returns the equation of the parallel line through point (x3, y3) given two points (x1, y1), (x2, y2).

  Args:
    x1: The x-coordinate of the first point.
    y1: The y-coordinate of the first point.
    x2: The x-coordinate of the second point.
    y2: The y-coordinate of the second point.
    x3: The x-coordinate of the point through which the parallel line passes.
    y3: The y-coordinate of the point through which the parallel line passes.

  Returns:
    The equation of the parallel line in the form y = mx + b.
  """
  (x1, y1), (x2, y2) = point1, point2
  # The slope of the first line is (y2 - y1) / (x2 - x1).
  m = (y2 - y1) / (x2 - x1 + 0.00001)

  # The equation of the parallel line is y = mx + b, where b is the y-intercept.
  # To find b, we can substitute the point (x3, y3) into the equation.
  b = point[1] - m * point[0]

  x_new_1 = int((point[1] - 800 - b)/(m+0.0001))
  y_new_1 = point[1] - 800

  x_new_2 = int((point[1] + 800 - b)/(m+0.0001))
  y_new_2 = point[1] + 800

  return (x_new_1, y_new_1), (x_new_2, y_new_2)


def to_the_left(point, point1, point2):
    """
    Determine whether the point is to the left of the line that comes through point1 and point2
    """
    return (point2[0] - point1[0])*(point[1] - point1[1]) - (point2[1] - point1[1])*(point[0] - point1[0]) > 0

def is_above(point, point1, point2):
    """
    Determine whether the point is to the top of the line that comes through point1 and point2
    """
    (x1, y1), (x2, y2) = (point1, point2)
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return point[1] < m * point[0] + b

def all_equal(arr):
    """
    Checks if all values in the array are equal (and not None)
    """
    value0 = arr[0]
    if value0 is not None:
        return np.all(arr == value0)
    else:
        return False


def line_intersection(line1, line2):
    """
    Find the intersection of two lines given by two points each
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    m1 = (y2 - y1) / (x2 - x1 + 0.00001)
    m2 = (y4 - y3) / (x4 - x3 + 0.00001)
    # Check if the lines are parallel
    if m1 == m2:
        return None  # No intersection
    else:
        x_intersect = (m1 * x1 - y1 - m2 * x3 + y3) / (m1 - m2 + 0.00001)
        y_intersect = m1 * (x_intersect - x1) + y1
        return (int(x_intersect), int(y_intersect))

def x_or_o_individual(pts, img):
	"""
	Determine X or O if could not be determined using contours
	Steps: 
	1. Select a window where X or O should be
	2. Perform a Perspective Transformation
	3. Convert to grayscale (I used Saturation channel of HSV instead of convertion to grayscale since it works well)
	4. Perform Otsu binarization
	5. Crop the image (since a window usually contains a part of a line of a grid)
	6. Find large enough contours
	7. Determine X or O based on characteristics of a contour
	"""
	width = 350
	height = 350

	def order_points(pts):
		# initialzie a list of coordinates that will be ordered
		# such that the first entry in the list is the top-left,
		# the second entry is the top-right, the third is the
		# bottom-right, and the fourth is the bottom-left
		rect = np.zeros((4, 2), dtype = "float32")
		# the top-left point will have the smallest sum, whereas
		# the bottom-right point will have the largest sum
		s = pts.sum(axis = 1)
		rect[0] = pts[np.argmin(s)]
		rect[2] = pts[np.argmax(s)]
		# now, compute the difference between the points, the
		# top-right point will have the smallest difference,
		# whereas the bottom-left will have the largest difference
		diff = np.diff(pts, axis = 1)
		rect[1] = pts[np.argmin(diff)]
		rect[3] = pts[np.argmax(diff)]
		# return the ordered coordinates
		return rect

	input = order_points(np.float32(pts))
	output = order_points(np.float32([[0,0], [width-1,0], [width-1,height-1], [0,height-1]]))

	# compute perspective matrix
	matrix = cv2.getPerspectiveTransform(input,output)
	imgOutput = cv2.warpPerspective(img, matrix, (width,height), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
	# Convert to grayscale
	imgOutHSV = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2HSV)
	gray = imgOutHSV[:, :, 1]
	# Otsu binarization
	_, gray_thresh = cv2.threshold(gray, 0, 255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# Crop
	thresh_croped = gray_thresh[25:-25, 25:-25] # removing existing border
	# Find contours
	contours1, hierarchy1 = cv2.findContours(thresh_croped, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	# Calculate contour parameters
	hull1 = []
	convexity_measure1 = []
	centers1 = []
	areas1 = []
	convex_areas1 = []
	relative_areas1 = []
	dists1 = []
	# Remove small contours
	contours1 = [cnt for cnt in contours1 if cv2.contourArea(cv2.convexHull(cnt, False)) / (width * height) > 0.02]
	# cv2.imshow("Perspective", thresh_croped)
	# cv2.waitKey(0) #wait for any key
	# cv2.destroyAllWindows()

	# calculate points for each contour
	for i, contour in enumerate(contours1):
		# creating convex hull object for each contour
		hull1.append(cv2.convexHull(contour, False))
		areas1.append(cv2.contourArea(contour))
		convex_areas1.append(cv2.contourArea(hull1[-1]))
		convexity_measure1.append(areas1[-1]/convex_areas1[-1])
		M = cv2.moments(contour)
		centers1.append((int(M['m10']/M['m00']), int(M['m01']/M['m00'])))
		relative_areas1.append(convex_areas1[-1] / (width * height))
		dists1.append(cv2.pointPolygonTest(contour, centers1[-1], True))
          
	# TODO cover more cases 
	if len(contours1) == 1:
		if convexity_measure1[i] > 0.5 and abs(dists1[i]) > 15:
			return 'o'
		
		elif convexity_measure1[i] < 0.5 and abs(dists1[i]) < 15:
			return 'x'
		
		elif abs(dists1[i]) > 20 and 0.1 < relative_areas1[i] < 0.7:
			return 'o'

	if len(contours1) == 2:
		if convexity_measure1[0] > 0.5 and convexity_measure1[1] > 0.5:
			return 'o' 	
		elif min(abs(dists1[0]), abs(dists1[1])) > 10:
			return 'o'
		
		concated = np.vstack([contours1[0], contours1[1]]) # Union of contours
		M = cv2.moments(concated)
		center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
		if abs(cv2.pointPolygonTest(concated, center, True)) > 15: # If union is circle-shaped then 'o'
			return 'o'
          


def tic_tac_toe(img_path):
    # Load the image
    img = cv2.imread(img_path)
    # Segment the image
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert to HSV
    segmented = skimage.segmentation.clear_border(img_hsv) # Perform segmentation
    # Binarize the image and remove the noise
    mask = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY) # The resulting segmenation is RGB, so converting to grayscale
    blurred_mask = cv2.GaussianBlur(mask, (7,7), 0) # Blur the grayscale image
    _, thresh = cv2.threshold(blurred_mask, 8, 255, cv2.THRESH_BINARY) # Binarize the image
    kernel = np.ones((3,3), np.uint8)
    thresh_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # Removing pepper
    thresh_morph = cv2.morphologyEx(thresh_close, cv2.MORPH_OPEN, kernel) # Removing salt
    # Find contours
    contours, hierarchy = cv2.findContours(thresh_morph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Find grid lines
    skeleton = skeletonize(thresh_morph / 255).astype(np.uint8) * 255 # Skeletonize the image (this way HoughLines works better)
    lines = cv2.HoughLines(skeleton, 1, np.pi/360, 100, 0, 0) # Find Hough lines
    # Non max supress (Need to find only 4 lines)
    grid_lines = [lines[0]]

    for r_theta in lines[1:]:
        r, theta = r_theta[0]
        flag = 0

        # For each line check whether it is close to existed grid lines and add if not
        for r_theta2 in grid_lines:
            r2, theta2 = r_theta2[0]
            # Close lines have both theta and rho similar. Next line checks whether it is the case, theta is close when the difference is 0 or PI.
            if (abs(theta2 - theta) < np.pi/18 or abs(np.pi - abs(theta2 - theta)) < np.pi/18) and abs(abs(r2) - abs(r)) < 60:    
                flag = 1
        # If line is not similar to existing ones, it is added to grid lines. We only look for 4 strongest lines, if we found 4 - break
        if flag == 0:
            grid_lines.append(r_theta)
            if len(grid_lines) > 3:
                break

    # Determine whether lines are vertical or horizontal
    vertical_lines = sorted([line for line in grid_lines if abs(line[0][1] - np.pi/2) >= 1], key=lambda x: abs(x[0][0])) # First is left
    horizontal_lines = sorted([line for line in grid_lines if abs(line[0][1] - np.pi/2) < 1], key=lambda x: abs(x[0][0])) # First is upper

    # Order is Upper Left, Lower Left, Upper Right, Lower Right
    intersections = [hough_inter(*horizontal_lines[0][0], *vertical_lines[0][0]), hough_inter(*horizontal_lines[1][0], *vertical_lines[0][0]),
                    hough_inter(*horizontal_lines[0][0], *vertical_lines[1][0]), hough_inter(*horizontal_lines[1][0], *vertical_lines[1][0])]


    # Intersection points
    point1, point2, point3, point4 = intersections # Order: Upper Left, Lower Left, Upper Right, Lower Right
    # Find the estimated width and height of a cell (take height and width of the center's cell with some margin)
    dist_multiplier = 1.2
    x_dist = dist_multiplier * max(np.linalg.norm(intersections[0] - intersections[2]), np.linalg.norm(intersections[1] - intersections[3]))
    y_dist = dist_multiplier * max(np.linalg.norm(intersections[0] - intersections[1]), np.linalg.norm(intersections[2] - intersections[3]))

    # Estimate the grid with lines (4 drawn lines and outer lines are estimated to be parallel to main lines)
    l_point = (int(point2[0] - x_dist), find_y_given_x(point2[0] - x_dist, point2, point4))
    r_point = (int(point3[0] + x_dist), find_y_given_x(point3[0] + x_dist, point1, point3))
    top_point = (find_x_given_y(point1[1] - y_dist, point1, point2), int(point1[1] - y_dist))
    bottom_point = (find_x_given_y(point2[1] + y_dist, point3, point4), int(point2[1] + y_dist))
    # Find outer lines
    left_line = parallel_line(l_point, point1, point2)
    right_line = parallel_line(r_point, point3, point4)
    top_line = parallel_line(top_point, point1, point3)
    bottom_line = parallel_line(bottom_point, point2, point4)

    # Detect X and O and their positions
    # create hull array for convex hull points
    hull = []
    convexity_measure = []
    centers = []
    areas = []
    convex_areas = []
    relative_areas = [] # Note! Depends on dist_multiplier # TODO remove dependency
    dists = []

    # calculate points for each contour
    for i, contour in enumerate(contours):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contour, False))
        areas.append(cv2.contourArea(contour))
        convex_areas.append(cv2.contourArea(hull[-1]))
        convexity_measure.append(areas[-1]/convex_areas[-1])
        M = cv2.moments(contour)
        centers.append((int(M['m10']/M['m00']), int(M['m01']/M['m00'])))
        relative_areas.append(convex_areas[-1] / (x_dist*y_dist))
        dists.append(cv2.pointPolygonTest(contour, centers[-1], True))

    # create an empty black image to draw contours on it
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
    
    # draw contours and hull points
    for i in range(len(contours)):
        color_contours = (0, 255, 0) if 0.05 < relative_areas[i] < 1 else (100, 100, 100) # gray for contours not suitable for X or O
        color = (255, 0, 0) if 0.05 < relative_areas[i] < 1 else (100, 100, 100) # gray for contours not suitable for X or O
        # draw ith contour
        cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color, 1, 8)

    # Assign contours to the corresponding cells (upper/middle/lower left/center/right)   
    ul, ml, dl = [], [], []
    # UL
    for i, c in enumerate(contours):
        if to_the_left(centers[i], point1, point2) and is_above(centers[i], point1, point3) \
        and not is_above(centers[i], *top_line) and not to_the_left(centers[i], *left_line):
            if 0.05 < relative_areas[i] < 1:
                ul.append(i)
    # ML
    for i, c in enumerate(contours):
        if to_the_left(centers[i], point1, point2) and is_above(centers[i], point2, point4) \
        and not is_above(centers[i], point1, point3) and not to_the_left(centers[i], *left_line):
            if 0.05 < relative_areas[i] < 1:
                ml.append(i)
    # DL
    for i, c in enumerate(contours):
        if to_the_left(centers[i], point1, point2) and not is_above(centers[i], point2, point4) \
        and is_above(centers[i], *bottom_line) and not to_the_left(centers[i], *left_line):
            if 0.05 < relative_areas[i] < 1:
                dl.append(i)

    uc, mc, dc = [], [], []
    # UC
    for i, c in enumerate(contours):
        if to_the_left(centers[i], point3, point4) and is_above(centers[i], point1, point3) \
        and not is_above(centers[i], *top_line) and not to_the_left(centers[i], point1, point2):
            if 0.05 < relative_areas[i] < 1:
                uc.append(i)
    # MC
    for i, c in enumerate(contours):
        if to_the_left(centers[i], point3, point4) and not is_above(centers[i], point1, point3) \
        and is_above(centers[i], point2, point4) and not to_the_left(centers[i], point1, point2):
            if 0.05 < relative_areas[i] < 1:
                if not (relative_areas[i] > 0.5 and convexity_measure[i] > 0.92):
                    mc.append(i)
    # DC
    for i, c in enumerate(contours):
        if to_the_left(centers[i], point3, point4) and not is_above(centers[i], point2, point4) \
        and is_above(centers[i], *bottom_line) and not to_the_left(centers[i], point1, point2):
            if 0.05 < relative_areas[i] < 1:
                dc.append(i)

    ur, mr, dr = [], [], []
    # UR
    for i, c in enumerate(contours):
        if not to_the_left(centers[i], point3, point4) and is_above(centers[i], point1, point3) \
        and not is_above(centers[i], *top_line) and  to_the_left(centers[i], *right_line):
            if 0.05 < relative_areas[i] < 1:
                ur.append(i)
    # MR
    for i, c in enumerate(contours):
        if not to_the_left(centers[i], point3, point4) and is_above(centers[i], point2, point4) \
        and not is_above(centers[i], point1, point3) and  to_the_left(centers[i], *right_line):
            if 0.05 < relative_areas[i] < 1:
                mr.append(i)
    # DR
    for i, c in enumerate(contours):
        if not to_the_left(centers[i], point3, point4) and not is_above(centers[i], point2, point4) \
        and is_above(centers[i], *bottom_line) and to_the_left(centers[i], *right_line):
            if 0.05 < relative_areas[i] < 1:
                dr.append(i)

    # Store corner points of each cell to cut and perspective transform it (if cannot determine using contours)
    ul_key_points = [line_intersection(top_line, left_line), line_intersection((point1, point3), left_line), line_intersection((point1, point2), top_line), point1]
    ml_key_points = [line_intersection((point1, point3), left_line), line_intersection((point2, point4), left_line), point1, point2]
    dl_key_points = [line_intersection((point2, point4), left_line), line_intersection(bottom_line, left_line), point2, line_intersection(bottom_line, (point1, point2))]
    uc_key_points = [line_intersection((point1, point2), top_line), point1, line_intersection(top_line, (point3, point4)), point3]
    mc_key_points = [point1, point2, point3, point4]
    dc_key_points = [point2, line_intersection((point1, point2), bottom_line), point4,  line_intersection(bottom_line, (point3, point4))]
    ur_key_points = [line_intersection((point3, point4), top_line), point3, line_intersection(right_line, top_line), line_intersection(right_line, (point1, point3))]
    mr_key_points = [point3, point4, line_intersection((point1, point3), right_line), line_intersection((point2, point4), right_line)]            
    dr_key_points = [point4, line_intersection((point3, point4), bottom_line), line_intersection((point2, point4), right_line), line_intersection(bottom_line, right_line)]
    
    
    def x_or_o(lst, key_points):
        """
        Determine whether contours stands for X or O (or None)
        For that mostly used:
        1. Relative area to an area of 1 grid cell
        2. Convexity measure (1 if convex, close to 0 if very concave) - O get high score and X get low score
        3. Distance from the center of mass to the closest point - large for O and small for X
        """
        if len(lst) == 1:
            i = lst[0]
            if abs(dists[i]) > 15 and relative_areas[i] < 0.6:
                return 'o'
            elif 0.1 < relative_areas[i] < 0.6 and 0.1 < convexity_measure[i] < 0.4:
                return 'x'
            
        elif len(lst) == 2:
            i1 = lst[0]
            i2 = lst[1]
            if 0.8 < convexity_measure[i1] and 0.8 < convexity_measure[i2] and (hierarchy[0][i1][3] == i2 or hierarchy[0][i2][3] == i1):
                return 'o'
            elif (0.8 < convexity_measure[i1] and dists[i1] > 15) or (0.8 < convexity_measure[i2] and dists[i2] > 15):
                return 'o'


        # If could not determine by contour:
        return x_or_o_individual(key_points, img)
    
    
    # Store 'x' and 'o' in an array
    game = np.array([
        [x_or_o(ul, ul_key_points), x_or_o(uc, uc_key_points), x_or_o(ur, ur_key_points)],
        [x_or_o(ml, ml_key_points), x_or_o(mc, mc_key_points), x_or_o(mr, mr_key_points)],
        [x_or_o(dl, dl_key_points), x_or_o(dc, dc_key_points), x_or_o(dr, dr_key_points)]
    ])

    combinations = [game[:, 0], game[:, 1], game[:, 2], game[0, :], game[1, :], game[2, :], np.diag(game), np.diag(np.fliplr(game))]
    winner = [all_equal(comb) for comb in combinations].index(True)


    corner_point_idx = [(ul, dl), (uc, dc), (ur, dr), (ul, ur), (ml, mr), (dl, dr), (ul, dr), (dl, ur)]

    cv2.line(img, centers[corner_point_idx[winner][0][0]], centers[corner_point_idx[winner][1][0]], (0,0,255), 3, cv2.LINE_AA)

    cv2.imwrite(img_path[:-4] + '_output.png', img)




tic_tac_toe('tic_images/tic_12.jpg')