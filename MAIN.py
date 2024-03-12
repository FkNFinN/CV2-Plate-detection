### https://github.com/ultralytics/yolov5 ###
# 65011467 Pichapa Hansatit 
# 65011619 Varis Sligupta
# 65011620 Varit Roongrtoekarnkha 👑
# 65011648 Yotin Limyotin

import torch
import cv2
import json
import time
import numpy as np
import requests
import pathlib
from PIL import Image, ImageDraw, ImageFont

Apikey = "BghsplRaCk6QRBZR5fX7krSjOdn1RS0w"
url_lpr = "https://api.aiforthai.in.th/lpr-v2"

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

detected_start = None
detected_duration = 0.5

mul_rate =2
padding = 20

shear_angle_y = 0.0
shear_angle_x = 0.0
shift_y = 0.0
shift_x = 0.0
zoom = 1
thres = 0

vid = cv2.VideoCapture(0)
model = torch.hub.load('.', 'custom', path='best.pt', source='local', device='mps') 

def plot_boxes(result_dict, frame):
    rec_st, rec_en = None, None
    for ob in result_dict:
        rec_st = (int(ob['xmin'] - padding), int(ob['ymin'] - padding))
        rec_en = (int(ob['xmax'] + padding), int(ob['ymax'] + padding))
        color = (255, 0, 0)
        thickness = 3
        cv2.rectangle(frame, rec_st, rec_en, color, thickness)
        cv2.putText(frame, "%s %0.2f" % (ob['name'], ob['confidence']), rec_st, cv2.FONT_HERSHEY_DUPLEX, 2, color, thickness)
    return frame, rec_st, rec_en

def update_shear_y(val):
    global shear_angle_y
    shear_angle_y = (val - 50)/80
    update_sheared_image()

def update_shear_x(val):
    global shear_angle_x
    shear_angle_x = (val - 50)/80
    update_sheared_image()

def update_shift_y(val):
    global shift_y
    shift_y = (val - 50)*3.5
    update_sheared_image()

def update_shift_x(val):
    global shift_x
    shift_x = (val - 50)*3.5
    update_sheared_image()

def update_zoom(val):
    global zoom
    zoom = val/100 + 1
    update_sheared_image()

def update_thres(val):
    global thres
    thres = int(val * 2.55)
    update_sheared_image()

def update_sheared_image():
    global image, shear_angle_y, shear_angle_x, sheared_image,thres
    shear_matrix = np.array([[zoom, np.tan(shear_angle_x), shift_x],
                               [np.tan(shear_angle_y), zoom, shift_y]], dtype=np.float32)
    sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))

    _, sheared_image = cv2.threshold(sheared_image, thres, 255, cv2.THRESH_BINARY)

    cv2.rectangle(sheared_image, (int(sheared_image.shape[1]/5), 0), (int(sheared_image.shape[1]) - int(sheared_image.shape[1]/5), int(sheared_image.shape[0]/2.5)), (255,0,0), 2)
    cv2.rectangle(sheared_image, (0, int(sheared_image.shape[0]) - int(sheared_image.shape[0]/2.5)), (int(sheared_image.shape[1]), int(sheared_image.shape[0])), (0,255,0), 2)
    cv2.imshow('Align', sheared_image)
    sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))

def shear_button_callback():
    global sheared_image, frame
    # cv2.imwrite('plate.jpg', sheared_image)
    combined_image = np.hstack((sheared_image[0:int(sheared_image.shape[0]/2.5),int(sheared_image.shape[1]/5):int(sheared_image.shape[1]) - int(sheared_image.shape[1]/5)], 
                                sheared_image[int(sheared_image.shape[0]) - int(sheared_image.shape[0]/2.5):int(sheared_image.shape[0])]))
    cv2.imwrite('swap.jpg', cv2.resize(combined_image, (4 * abs(combined_image.shape[1]), 4 * abs(combined_image.shape[0]))))

    payload = {'crop': '0', 'rotate': '0'}
    files = {'image':open('swap.jpg', 'rb')}
    headers = {
        'Apikey': Apikey,
        }

    response = requests.post(url_lpr, files=files, data = payload, headers=headers)
    text = np.array([item.get('lpr') for item in response.json()])

    l_img = cv2.imread("Form.jpg")
    s_img = frame
    s_img = cv2.resize(s_img,(320, 240))

    l_img[270:270+s_img.shape[0], 80:80+s_img.shape[1]] = s_img
    imgpil = Image.fromarray(l_img)
    draw = ImageDraw.Draw(imgpil)

    draw.text((230, 200),  str(text[0]), font = ImageFont.truetype('ANGSA.ttf', 20), fill = (0,0,0,0))
    draw.text((130, 175),  str(time.strftime("%d/%m/%Y", time.localtime())), font = ImageFont.truetype('ANGSA.ttf', 18), fill = (0,0,0,0))
    draw.text((250, 175),  str(time.strftime("%H:%M", time.localtime())), font = ImageFont.truetype('ANGSA.ttf', 18), fill = (0,0,0,0))
    draw.text((278, 175),  str('น'), font = ImageFont.truetype('ANGSA.ttf', 18), fill = (0,0,0,0))

    res = np.array(imgpil)
    cv2.imshow('Form', res)
    cv2.imwrite('Fine.jpg', res)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('w'):
            tokenline = 'jbwYqshJvjD3LnCrwLSBs3sBo828310l8JYh0LVdifg'
            headers = {'Authorization':'Bearer ' + tokenline}
            payload = {"message": text[0]}
            files = {'imageFile': open('Fine.jpg', 'rb')}
            response = requests.post("https://notify-api.line.me/api/notify", headers=headers, data=payload, files=files)
        elif key == ord('s'):
            shear_button_callback()
            break
        elif key == ord('q'):
            cv2.destroyWindow('Form')
            break

def main():
    global sheared_image, frame, image, detected_start,shear_angle_y, shear_angle_x, shift_y, shift_x, zoom
    while True:
        _ , frame = vid.read()
        frame_canny = cv2.Canny(frame, 150, 250)
        
        results = model(frame_canny)
        result_jsons = results.pandas().xyxy[0].to_json(orient="records")
        result_dict = json.loads(result_jsons)
        dframe = frame.copy()

        _ , rec_start, rec_end = plot_boxes(result_dict, dframe)
        cv2.imshow('RAW', dframe)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        plate_detected = any(ob['name'] == 'plate' for ob in result_dict)

        if plate_detected:
            if detected_start is None:
                detected_start = time.time()
            else:
                if time.time() - detected_start >= detected_duration:
                    cropped_region = cv2.resize(frame[rec_start[1]:rec_end[1], rec_start[0]:rec_end[0]],
                                                        (mul_rate * abs(rec_start[0] - rec_end[0]), 
                                                        mul_rate * abs(rec_start[1] - rec_end[1])))
                    image = cropped_region
                    update_sheared_image()

                    cv2.namedWindow('Shear Transformation', cv2.WINDOW_NORMAL)
                    cv2.createTrackbar('Shear Y', 'Shear Transformation', 0, 100, update_shear_y)
                    cv2.setTrackbarPos('Shear Y', 'Shear Transformation', 50)
                    cv2.createTrackbar('Shear X', 'Shear Transformation', 0, 100, update_shear_x)
                    cv2.setTrackbarPos('Shear X', 'Shear Transformation', 50)
                    cv2.createTrackbar('Y Shift', 'Shear Transformation', 0, 100, update_shift_y)
                    cv2.setTrackbarPos('Y Shift', 'Shear Transformation', 50)
                    cv2.createTrackbar('X Shift', 'Shear Transformation', 0, 100, update_shift_x)
                    cv2.setTrackbarPos('X Shift', 'Shear Transformation', 50)
                    cv2.createTrackbar('Zoom', 'Shear Transformation', 0, 100, update_zoom)
                    cv2.createTrackbar('Threshold', 'Shear Transformation', 0, 100, update_thres)
                    cv2.setTrackbarPos('Threshold', 'Shear Transformation', 90)
                    cv2.imshow('Crop2', cv2.resize(frame[rec_start[1]:rec_end[1], rec_start[0]:rec_end[0]], (mul_rate * abs(rec_start[0] - rec_end[0]), mul_rate * abs(rec_start[1] - rec_end[1]))))
                    cv2.imwrite('RAW.jpg',frame)
                    # cv2.imshow('Canny with Boxes', frame_canny_with_boxes)
                    while True:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('s'):
                            shear_button_callback()
                        elif key == ord('q'):
                            cv2.destroyWindow('Shear Transformation')
                            break
                    shear_angle_y = 0.0
                    shear_angle_x = 0.0
                    shift_y = 0.0
                    shift_x = 0.0
                    zoom = 1
                    detected_start = None

    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()