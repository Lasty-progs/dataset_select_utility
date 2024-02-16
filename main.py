import cv2
import glob
import numpy as np
import math
folder_path = 'images/'

image_files = glob.glob("./" + folder_path + "*.png")
selected_files = []

n,m=4,4
w,h = 1920,1080
w_img,h_img = w//n, h//m

def render_batch(files):
    pos2file=[]
    window = np.zeros((h,w,3),dtype = np.uint8)

    for i,file in enumerate(files):
        x,y=i%m,i//n
        img = cv2.imread(file)
        window[y*h_img:(y+1)*h_img,x*w_img:(x+1)*w_img] = cv2.resize(img,(w_img,h_img), interpolation = cv2.INTER_CUBIC)
        if i%m==0:
            pos2file.append([])
        pos2file[-1].append(file)
    return window, pos2file


# function for operate mouse events
def mouse_callback(event, x, y, flags, current_selected):
    
    if flags!=1 and flags !=16 and flags !=17:
        return 
    xi, yi = x//w_img,y//h_img
    x_centre, y_centre = xi*w_img+w_img//2, yi*h_img+h_img//2
    if flags == 1:
        if yi < len(pos2file) and xi < len(pos2file[yi]):
            current_selected[yi,xi]=True
            cv2.circle(img,(x_centre,y_centre),10,(0,0,255),-1)
    elif (flags == 17 or flags == 16):
        if yi < len(pos2file) and xi < len(pos2file[yi]):
            current_selected[yi,xi]=False
            cv2.circle(img,(x_centre,y_centre),10,(0,255,255),-1)

    cv2.imshow('Image', img) 

def save_images(files):
    with open('Excluded.txt', 'w') as out_file:
        for file in files:
            out_file.write(file+"\n")

batch_size = n*m
for batch_i in range(math.ceil(len(image_files)/batch_size)):
    batch = image_files[batch_i*batch_size: (batch_i+1)*batch_size]
    img, pos2file = render_batch(batch)

    current_selected = np.zeros((n,m), dtype = np.bool_)
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', mouse_callback, current_selected)
    key = cv2.waitKey(0)
        
    if key == 32:  # push space for close window
        for i in range(len(current_selected)):
            for j in range(len(current_selected[i])):
                if current_selected[i][j]:
                    selected_files.append(pos2file[i][j])

        cv2.destroyAllWindows()
        continue
    if key == 27: # push esc for quick save and exit

        cv2.destroyAllWindows()
        break


save_images(selected_files)
