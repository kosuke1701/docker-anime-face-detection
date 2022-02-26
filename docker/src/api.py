import os
import sys
import _pickle as pic

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from fastapi import FastAPI, File, UploadFile

from face_pose import FacePose
from face_cluster import FaceCluster

detector = FacePose()
fc = pic.load(open("fc.bin", "rb"))

app = FastAPI()

@app.post("/detect_faces")
def detect_faces(file: UploadFile):
    img_arr = file.file.read()

    infos, img = detector.load_image(img_arr)
    infos = [detector.convert_ndarray_to_list(info) for info in infos]

    for i in range(len(infos)):
        fc_out = fc.predict(infos[i])
        fc_out["valid_score"] = float(fc_out["valid_score"])
        fc_out["origin"] = fc_out["origin"].tolist()
        fc_out["scaleX"] = float(fc_out["scaleX"])
        fc_out["scaleY"] = float(fc_out["scaleY"])
        fc_out["clusters"] = {key: int(val) for key, val in fc_out["clusters"].items()}
        fc_out["features"] = {key: val.tolist() for key, val in fc_out["features"].items()}

        infos[i].update(fc_out)
        infos[i]["valid_score_threshold"] = fc.anomaly_threshold

    return infos
