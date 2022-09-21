# Inference for ONNX model
from cgi import print_arguments
import cv2
import random
import numpy as np
import onnxruntime as ort
import time


cuda = True
w = "best.onnx"
cap = cv2.VideoCapture('vid_in2.mp4')
providers = ['CUDAExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)


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


# Pre-defined classes 
names = ['blue_cone', 'large_orange_cone', 'orange_cone', 'yellow_cone']
# names = ['blue_small_case','black_large_tophandle','orange_small_case','grey_large_tophandle','green_small_bag','green_large_ammobox','black_large_crate','blue_small_bag']
# names = ['crate', 'case', 'bag']

# Allocate random colours for each bounding box
colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}


# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0

# detect objects in each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
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
        



        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))
        cv2.putText(ori_images[0],"FPS: {0}".format(fps),(5, 25),cv2.FONT_HERSHEY_SIMPLEX,0.75,[0, 255, 0],thickness=2)  

        # loop over each of the detections
        for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
            # print("batch id:{0}".format(batch_id))
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
            # print(box[:2] + [w,h])
            cv2.rectangle(image,box[:2],box[2:],color,2)
            cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)  

 
        # print(list(ori_images[0].shape[:2]))
        cv2.imshow('show', ori_images[0])
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()