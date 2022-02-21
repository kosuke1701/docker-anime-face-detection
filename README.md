

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

# Return (OpenCV's coordinate system is used):
[
    {
        "bbox": [xmin, ymin, xmax, ymax],
        "keypoints": [[x,y]*(28 landmarks)],
        "angle": (face rotation angle),
        "center": [xcenter,ycenter],
        "rot_keypoints": [[x,y]*(28 aligned landmarks)]
    }
]
```