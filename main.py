#import libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2

from moviepy.editor import VideoFileClip
from cam_calibration import distortion_factors
from perspective_transform import warp
from binary_threshold import binary_thresholded
from Lane_Line_detect import find_lane_pixels_using_histogram, fit_poly, draw_poly_lines
from lane_info import project_lane_info
from vehicle_pos import measure_curvature_meters, measure_position_meters
from polyfit import find_lane_pixels_using_prev_poly
from pipeline import lane_finding_pipeline

### STEP 3: Process Binary Thresholded Images ###

img = cv2.imread('test_images/test4.jpg')
plt.imshow(img)
plt.show()

binary_thresh = binary_thresholded(img)
out_img = np.dstack((binary_thresh, binary_thresh, binary_thresh))*255

plt.imshow(out_img)
plt.show()


img = cv2.imread('test_images/test6.jpg')

binary_thresh = binary_thresholded(img)
out_img = np.dstack((binary_thresh, binary_thresh, binary_thresh))*255
binary_warped, M_inv = warp(binary_thresh)
plt.imshow(out_img, cmap='gray')
plt.show()


leftx, lefty, rightx, righty = find_lane_pixels_using_histogram(binary_warped)
left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped,leftx, lefty, rightx, righty)
# print(left_fit)
out_img = draw_poly_lines(binary_warped, left_fitx, right_fitx, ploty)
plt.imshow(out_img)
plt.show()


prev_left_fit, prev_right_fit = left_fit, right_fit
# print('prev left fit: ', prev_left_fit)
# print('prev right fit: ', prev_right_fit)

### STEP 5: Detection of Lane Lines Based on Previous Step ###

img = cv2.imread('test_images/test5.jpg')
binary_thresh = binary_thresholded(img)
binary_warped, M_inv = warp(binary_thresh)

# Polynomial fit values from the previous frame
# Make sure to grab the actual values from the previous step in your project!
#left_fit = np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02])
#right_fit = np.array([4.17622148e-04, -4.93848953e-01,  1.11806170e+03])


leftx, lefty, rightx, righty = find_lane_pixels_using_prev_poly(binary_warped,prev_left_fit, prev_right_fit)
left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped,leftx, lefty, rightx, righty)
out_img = draw_poly_lines(binary_warped, left_fitx, right_fitx, ploty)
plt.imshow(out_img)
plt.show()

### STEP 6: Calculate Vehicle Position and Curve Radius ###

left_curverad, right_curverad =  measure_curvature_meters(binary_warped, left_fitx, right_fitx, ploty)
print('left curve radius in meters  = ', left_curverad)
print('right curve radius in meters = ', right_curverad)
veh_pos = measure_position_meters(binary_warped, left_fit, right_fit)
print('vehicle position relative to center  = ', veh_pos)

### STEP 7: Project Lane Delimitations Back on Image Plane and Add Text for Lane Info ###

new_img = project_lane_info(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), binary_warped, ploty, left_fitx, right_fitx, M_inv, left_curverad, right_curverad, veh_pos)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image', fontsize=20)
ax2.imshow(new_img, cmap='gray')
ax2.set_title('Image With Lane Marked', fontsize=20)


### STEP 8: Lane Finding Pipeline on Video ###

input_file = "project_video.mp4" #input file (keep in same dir of this project or change path
output_file = 'project_video_output.mp4' # video output file
clip1 = VideoFileClip(input_file)

output_clip = clip1.fl_image(lane_finding_pipeline)  #fl_image accepts image function as parameter
output_clip.write_videofile(output_file, audio=False) # write to output file
