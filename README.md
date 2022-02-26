

## Getting Started
### Prerequisite

Following softwares are required to build the docker image.

* nvidia-docker
  - https://github.com/NVIDIA/nvidia-docker
  - Edit docker configuration to change default runtime to `nvidia`.
    - ```
        # E.g. edit /etc/docker/daemon.json as follows.
        {
            "default-runtime": "nvidia",
            "runtimes": {
                "nvidia": {
                    "path": "nvidia-container-runtime",
                    "runtimeArgs": []
                }
            }
        }
        ```
* nvidia driver compatible to CUDA 11.3

### Building Docker Image

```
docker build -t face_pose docker
```

### Start API

```
docker run --gpus all -p 32011:32011 -t face_pose uvicorn api:app --host 0.0.0.0 --port 32011
```

After API is started, you can access API as follows:

```
curl -X POST -F file=@/path/to/image http://localhost:32011/detect_faces
```

Returned json data is a list which contains a dictionary for each detected face with following keys. (OpenCV's coordinate system is used.)

* "bbox"
  - bounding box of a detected face
  - [xmin, ymin, xmax, ymax]
* "keypoints"
  - 28 landmarks points of a detected face (see https://github.com/hysts/anime-face-detector)
* "angle"
  - face rotation angle
* "valid_score"
  - If this value is low, it is more likely that detected landmark points are anomalous.
* "valid_score_threshold"
  - Default threshold for values of "valid_score".
* "features"
  - Normalized and flattened landmark positions for each face components.
* "origin", "scaleX", "scaleY"
  - Values used to transform landmark positions when computing values of "features".