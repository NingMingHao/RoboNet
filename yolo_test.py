
from yolo_tiny import YOLO
from PIL import Image
import cv2
import numpy as np
from timeit import default_timer as timer

# Blue
# video_input_path = '/Users/minghao/Documents/University/Robomaster/Object_Detection/raw_data/2020_1_4_15_27_47_2.mp4'

# Red
# video_input_path = '/Users/minghao/Documents/University/Robomaster/Object_Detection/raw_data/2020_1_4_15_40_43_178.mp4'

# Data Unseen
# video_input_path = '/Users/minghao/Documents/University/Robomaster/Object_Detection/raw_data/2020_1_4_16_1_45_967.mp4'

# video_input_path = '/Users/minghao/Documents/University/Robomaster/Object_Detection/raw_data_3_cars/2020_1_14_15_38_16_374.avi'
video_input_path = '/Users/minghao/Documents/University/Robomaster/Object_Detection/raw_data_4_cars/2020_1_16_17_43_13_574.mp4'


video_output_path = ""

is_to_detect_image = False


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            result = np.asarray(r_image)
            result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
            cv2.imshow('result', result)
            cv2.waitKey(1)
            
    yolo.close_session()



def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray( cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) )
        image, bbox_list = yolo.detect_image(image)
        result = np.asarray(image)
        result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()
    

if __name__ == '__main__':
    if is_to_detect_image:
        detect_img(YOLO())
    else:
        detect_video(YOLO(), video_input_path, video_output_path)
