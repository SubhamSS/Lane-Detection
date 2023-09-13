import numpy as np
from binary_threshold import binary_thresholded
from perspective_transform import warp
from polyfit import find_lane_pixels_using_prev_poly
from Lane_Line_detect import fit_poly, find_lane_pixels_using_histogram
from vehicle_pos import measure_curvature_meters, measure_position_meters
from lane_info import project_lane_info

def lane_finding_pipeline(img):
    left_fit_hist = np.array([])
    right_fit_hist = np.array([])
    binary_thresh = binary_thresholded(img)
    binary_warped, M_inv = warp(binary_thresh)
    # out_img = np.dstack((binary_thresh, binary_thresh, binary_thresh))*255
    if (len(left_fit_hist) == 0):
        leftx, lefty, rightx, righty = find_lane_pixels_using_histogram(binary_warped)
        left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped, leftx, lefty, rightx, righty)

        # Store fit in history
        left_fit_hist = np.array(left_fit)
        new_left_fit = np.array(left_fit)
        left_fit_hist = np.vstack([left_fit_hist, new_left_fit])
        right_fit_hist = np.array(right_fit)
        new_right_fit = np.array(right_fit)
        right_fit_hist = np.vstack([right_fit_hist, new_right_fit])
    else:
        prev_left_fit = [np.mean(left_fit_hist[:, 0]), np.mean(left_fit_hist[:, 1]), np.mean(left_fit_hist[:, 2])]
        prev_right_fit = [np.mean(right_fit_hist[:, 0]), np.mean(right_fit_hist[:, 1]), np.mean(right_fit_hist[:, 2])]
        leftx, lefty, rightx, righty = find_lane_pixels_using_prev_poly(binary_warped,prev_left_fit,prev_right_fit)
        if (len(lefty) == 0 or len(righty) == 0):
            leftx, lefty, rightx, righty = find_lane_pixels_using_histogram(binary_warped)
        left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped, leftx, lefty, rightx, righty)

        # Add new values to history
        new_left_fit = np.array(left_fit)
        left_fit_hist = np.vstack([left_fit_hist, new_left_fit])
        new_right_fit = np.array(right_fit)
        right_fit_hist = np.vstack([right_fit_hist, new_right_fit])

        # Remove old values from history
        if (len(left_fit_hist) > 10):
            left_fit_hist = np.delete(left_fit_hist, 0, 0)
            right_fit_hist = np.delete(right_fit_hist, 0, 0)

    left_curverad, right_curverad = measure_curvature_meters(binary_warped, left_fitx, right_fitx, ploty)
    # measure_curvature_meters(binary_warped, left_fitx, right_fitx, ploty)
    veh_pos = measure_position_meters(binary_warped, left_fit, right_fit)
    out_img = project_lane_info(img, binary_warped, ploty, left_fitx, right_fitx, M_inv, left_curverad, right_curverad,
                                veh_pos)
    return out_img
