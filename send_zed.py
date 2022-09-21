########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    Open the camera and start streaming images using H264 codec
"""
import sys
import pyzed.sl as sl
import cv2
import numpy as np
import onnxruntime as ort
import time
import random

w = "best.onnx"
session = ort.InferenceSession(w, providers=['CUDAExecutionProvider'])
names = ['blue_cone', 'large_orange_cone', 'orange_cone', 'yellow_cone', 'unknown_cone']
colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}
# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = time.time()

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def inference(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255
    im.shape

    outname = [i.name for i in session.get_outputs()]
    # outname

    inname = [i.name for i in session.get_inputs()]
    # inname

    inp = {inname[0]:im}
    # ONNX inference
    outputs = session.run(outname, inp)[0]
    # outputs
    ori_images = [frame.copy()]

    # new_frame_time = time.time()
    # fps = 1/(new_frame_time-prev_frame_time)
    # prev_frame_time = new_frame_time
    # fps = str(int(fps))
    # cv2.putText(ori_images[0],"FPS: {0}".format(fps),(5, 25),cv2.FONT_HERSHEY_SIMPLEX,0.75,[0, 255, 0],thickness=2)  
    bounding_dict_arr = []
    # loop over each of the detections
    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
        image = ori_images[int(batch_id)]
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score),3)
        name = names[cls_id]
        color = colors[name]
        name += ' '+str(score)
        w = box[2] - box[0]
        h = box[3] - box[1]
        cv2.rectangle(image,box[:2],box[2:],color,2)
        cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2) 
        bounding_dict = {"box" : box, "name" : name}
        bounding_dict_arr.append(bounding_dict)

    return ori_images[0], bounding_dict_arr
    # return ori_images[0]

def main():

    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD2K
    init.depth_mode = sl.DEPTH_MODE.QUALITY
    init.coordinate_units  = sl.UNIT.METER
    init.depth_maximum_distance = 40
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)
    
    runtime = sl.RuntimeParameters()
    mat_i = sl.Mat()
    mat_d = sl.Mat()
    mat_d2 = sl.Mat()

    key = ''
    print("  Quit : CTRL+C\n")
    while key != 113:
        err = cam.grab(runtime)
        if (err == sl.ERROR_CODE.SUCCESS) :
            cam.retrieve_image(mat_d, sl.VIEW.DEPTH)
            cam.retrieve_image(mat_i, sl.VIEW.LEFT)
            cam.retrieve_measure(mat_d2, sl.MEASURE.DEPTH) # Retrieve depth

            
            inferenced, bounding_dict_arr = inference(mat_i.get_data())
            depth = mat_d.get_data()
            
            # inferenced = ligma(mat_i.get_data())
            
            for i in bounding_dict_arr:
                cv2.rectangle(depth,i['box'][:2],i['box'][2:],(0,255,0),2)
                centre_x = (i['box'][0] + i['box'][2])/2
                centre_y = (i['box'][1] + i['box'][3])/2
                depth_value = mat_d2.get_value(centre_x,centre_y)
                depth_label = "{0} distance: {1} metres".format(i['name'], round(depth_value[1],3))
                cv2.putText(depth,depth_label,(i['box'][0], i['box'][1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2) 
                
            
            # cv2.imshow("ZED_depth", mat_d.get_data())
            # cv2.imshow("ZED_cam", mat_i.get_data())
            cv2.imshow("ZED_inferenced", inferenced)
            cv2.imshow("ZED_depth_inferenced", depth)
            key = cv2.waitKey(1)
        else :
            key = cv2.waitKey(1)

    cam.close()

if __name__ == "__main__":
    main()