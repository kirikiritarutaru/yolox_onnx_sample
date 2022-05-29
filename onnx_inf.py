import cv2
import numpy as np
import onnxruntime
from box import Box

from utils import (COCO_CLASSES, demo_postprocess,
                   multiclass_nms_class_agnostic, preproc, vis)

config = Box({
    "img_size": 416,
    "model_path": 'models/yolox_nano.onnx',
    "score_thr": 0.3,
})

config = Box({
    "img_size": 640,
    "model_path": 'models/yolox_s.onnx',
    "score_thr": 0.3,
})

# config = Box({
#     "img_size": 640,
#     "model_path": 'models/yolox_l.onnx',
#     "score_thr": 0.3,
# })


if __name__ == '__main__':
    input_shape = (config.img_size, config.img_size)
    cap = cv2.VideoCapture(0)
    session = onnxruntime.InferenceSession(
        config.model_path,
        # providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        providers=['CPUExecutionProvider']
    )

    while True:
        _, origin_img = cap.read()
        img, ratio = preproc(origin_img, input_shape)
        ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
        output = session.run(None, ort_inputs)
        predictions = demo_postprocess(output[0], input_shape)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms_class_agnostic(
            boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1
        )
        if dets is not None:
            final_boxes = dets[:, :4]
            final_scores = dets[:, 4]
            final_cls_inds = dets[:, 5]
            origin_img = vis(
                origin_img, final_boxes, final_scores, final_cls_inds,
                conf=config.score_thr, class_names=COCO_CLASSES
            )

        cv2.imshow('img', origin_img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()
