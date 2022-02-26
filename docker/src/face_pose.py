from copy import deepcopy
import logging

import cv2
import numpy as np

from anime_face_detector import create_detector

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FacePose:
    def __init__(self):
        self.detector = create_detector("yolov3")
        logger.info("Done loading face pose detector")
    
    def _align_face_pose(self, face_info):
        """
        Rotate keypoints so that both eyes becomes horizontal.
        """
        minX, minY, maxX, maxY, _ = face_info["bbox"]
        center = np.mean([[minX, minY], [maxX, maxY]], axis=0)[np.newaxis,:]
        keypoints = face_info["keypoints"]

        right_indexes = [11, 12, 13, 14, 15, 16]
        left_indexes = [17, 18, 19, 20, 21, 22]
        # +--->X
        # |
        # v
        # Y
        right_pos = np.mean(keypoints[right_indexes], axis=0)[:2]
        left_pos = np.mean(keypoints[left_indexes], axis=0)[:2]

        cos, sin = (left_pos - right_pos) / np.linalg.norm(left_pos - right_pos)

        face_info["angle"] = float(np.arctan2(sin, cos))
        face_info["center"] = center
        face_info["bbox"] = face_info["bbox"][:-1]
        face_info["keypoints"] = face_info["keypoints"][:,:-1]
        return face_info
    
    def convert_ndarray_to_list(self, info):
        new_info = {}
        for key, val in info.items():
            if isinstance(val, np.ndarray):
                val = val.tolist()
            new_info[key] = val
        return new_info 
    
    def load_image(self, fn):
        if isinstance(fn, str):
            img = cv2.imread(fn)
        elif isinstance(fn, bytes):
            img = cv2.imdecode(np.asarray(bytearray(fn), dtype=np.uint8), 3)
        else:
            raise Exception(f"Illegal argument type:\t{type(fn)}\t{fn}")
        infos = self.detector(img)
        infos = [self._align_face_pose(info) for info in infos]
        return infos, img

    def visualize_faces(self, infos, img):
        img = deepcopy(img)

        for info in infos:
            # points
            for x, y in info["keypoints"]:
                img = cv2.circle(img, (int(x),int(y)), 2, (0,0,255), -1)
            # aligned rectangle
            minX, minY, maxX, maxY = info["bbox"]
            vertex = np.array([
                [minX, minY],
                [maxX, minY],
                [maxX, maxY],
                [minX, maxY]
            ])
            angle = info["angle"]
            rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            vertex = np.dot(vertex - info["center"], rot_mat.T) + info["center"]
            vertex = vertex.astype(np.int32)
            img = cv2.polylines(img, [vertex], True, (0,255,0))
        
        return img

if __name__=="__main__":
    detector = FacePose()
    infos, img = detector.load_image(open("/data/danbooru_all_512px/all_512px/0589/552589.png", "rb").read())
    img = detector.visualize_faces(infos, img)
    cv2.imwrite("/data/tmp.png", img)