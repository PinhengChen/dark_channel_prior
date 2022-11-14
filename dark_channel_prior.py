""""
Dark Channel Prior (DCP)
Coded by Pinheng Chen (Pion Chen)
Last edited time: 2022-11-14
Jupyter Notebook: 
"""
import cv2
import numpy as np
import os

def get_dark_channel(img, k_size = 7):
    """
    img is the input image.
    k_size is the kernel size to set the center value as the minimum value in the kernel.
    
    returen Dark Channel.
    """
    rows, cols, channels = img.shape
    img_tmp = np.zeros((rows, cols, channels))
    img_dc = np.zeros((rows, cols))
    center_pos = int(k_size / 2.0)
    # Filter the min_value in a k_size kernel
    for ch in range (0, channels):
        for row in range (0, rows - k_size):
            for col in range (0, cols - k_size):
                min_value = img[row, col, ch]
                for k_row in range (0, k_size):
                    for k_col in range (0, k_size):
                        if img[row + k_row, col + k_col, ch] < min_value:
                            min_value = img[row + k_row, col + k_col, ch]
                img_tmp[row + center_pos, col + center_pos, ch] = min_value

    # Get the minimum value among RGB channels
    for row in range (0, rows):
        for col in range (0, cols):
            min_value = img_tmp[row, col, 0]
            if img[row, col, 1] < min_value:
                min_value = img_tmp[row, col, 1]
            if img[row, col, 2] < min_value:
                min_value = img_tmp[row, col, 2]
            img_dc[row, col] = min_value
            
    return img_dc

def get_atmospheric_light(img, img_dc):
    """
    img is the original image.
    img_dc is the Dark Channel.
    
    return the atmospheric light A.
    """
    lightest_list = [] # store in the format of [ {'position': (row, col), 'value': 255}, { , } ]
    original_pixels = []
    dc_tmp = img_dc.copy()
    lightest_num = int(0.001 * img_dc.size)
    
    rows, cols, channels = img.shape
    A_tmp = 0  # atmospheric light A: select the lightest pixel in the original image as the value of A
    
    while len(lightest_list) <= lightest_num:
        max_value = 0
        # find the max value in Dark Channel
        for row in range (0, rows):
            for col in range (0, cols):
                if dc_tmp[row, col] > max_value:
                    max_value = dc_tmp[row, col]
                    max_row, max_col = row, col
        dc_tmp[max_row, max_col] = 0

        # find the max intensity in original image
        for row in range (0, rows):
            if len(lightest_list) > lightest_num: # more than 0.1%
                break
            for col in range (0, cols):
                if len(lightest_list) > lightest_num: # more than 0.1%
                    break
                if img_dc[row, col] == max_value:
                    lightest_list.append({'position':(row, col), 'value':img_dc[row, col]})
                    original_pixels.append((img[row, col, 0], img[row, col, 1], img[row, col, 2]))
                    dc_tmp[row, col] = 0
                    # calculate the mean intensity of 3 channels in order to get A
                    r_value, g_value, b_value = int(img[row, col, 0]), int(img[row, col, 1]), int(img[row, col, 2])
                    mean_intensity = (r_value + g_value + b_value) / 3.0
                    if mean_intensity > A_tmp:
                        A_tmp = mean_intensity
                        A_value = img[row, col]
    
    A = np.zeros((rows, cols, channels))
    for row in range (0, rows):
        for col in range (0, cols):
            for ch in range (0, channels):
                A[row, col, ch] = A_value[ch]
    
    return A

# From: https://blog.csdn.net/wsp_1138886114/article/details/84228939
def guideFilter(p, i, r, e = 0.0001):
    """
    The function of this filter is smooth the image according to a guide image.
    p is the guide image.
    i is the original image.
    r is the window/kernel size.
    e is a backup Denominator, which is used to avoid the situation that the denominator become 0.
    
    return a guide filtered image q.
    """ 
    mean_I = cv2.boxFilter(i, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    # 3
    a = cov_Ip / (var_I + e)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * i + mean_b
    return q

def get_transmission(img, A, omega = 0.95, k_size = 7):
    """
    img is the original image.
    A is the atmospheric light.
    omega is a constant parameter, it is used to keep tiny fog in the image. We use 0.95 by default.
    t_thres is the threshold of t. If t < t_thres, then t = t_thres. In order to avoid making the image too light.
    
    return the transmission t.
    """
    rows, cols, channels = img.shape
    
    # get img/A
    img_A = img / A
    # get min(min(img/A))
    img_A_dc = get_dark_channel(img_A, k_size)
    # guide filtering
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float64')
    img_A_dc = guideFilter(img_A_dc, img_grey , 225, 0.0001)
    # get t
    t = 1 - omega * img_A_dc
    
    return t

def remove_fog(img, t_thres = 0.1, omega = 0.95, k_size = 7):
    """
    img is the original image.
    A is the Atmosperic Light.
    t is the Transmission.
    t_thres is the threshold of t. If t < t_thres, then t = t_thres. In order to avoid making the image too light.
    
    return an image img_fogremoved whose fog has been removed.
    """
    rows, cols, channels = img.shape
    img_fogremoved = np.zeros((rows, cols, channels))
    
    # get dark_channel
    img_dc = get_dark_channel(img, k_size)
    # get A
    A = get_atmospheric_light(img, img_dc)
    # get t
    t = get_transmission(img, A, omega, k_size)
    # get the fog removed image
    for row in range(0, rows):
        for col in range(0, cols):
            for ch in range(0, channels):
                img_fogremoved[row, col, ch] = (img[row, col, ch] - A[row, col, ch]) \
                                             / (max(t[row, col], t_thres)) \
                                             + A[row, col, ch]
                if img_fogremoved[row, col, ch] < 0:
                    img_fogremoved[row, col, ch] = 0
                elif img_fogremoved[row, col, ch] > 255:
                    img_fogremoved[row, col, ch] = 255
                elif img_fogremoved[row, col, ch] >= 0 and img_fogremoved[row, col, ch] <= 255:
                    img_fogremoved[row, col, ch] = int(img_fogremoved[row, col, ch])

    return np.uint8(img_fogremoved)




if __name__ == '__main__':
    num_current = 0
    cwd = os.getcwd()
    folder = '/dataset/selected_images/'
    out_folder = '/dataset/dcp/'
    for filename in os.listdir(cwd + folder):
        num_current = num_current + 1
        file = cwd + folder + filename
        image_origin = cv2.imread(file)
        result = remove_fog(image_origin, k_size = 3)
        print("[{a}]  {b} Processed!".format(a = num_current, b = filename))
        cv2.imwrite(cwd + out_folder + filename, result)