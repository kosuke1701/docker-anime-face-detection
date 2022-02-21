from fastapi import FastAPI, File, UploadFile

from face_pose import FacePose

detector = FacePose()

app = FastAPI()

@app.post("/detect_faces")
def detect_faces(file: UploadFile):
    img_arr = file.file.read()

    infos, img = detector.load_image(img_arr)
    infos = [detector.convert_ndarray_to_list(info) for info in infos]

    return infos
