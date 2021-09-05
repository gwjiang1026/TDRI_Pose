
# How to build image
```
## build docker image
image_name="tdri/mediapipe-python-flask:0_0_1"
docker build -t "$image_name" .
```
# Run Container 
### run as deamon 
```
container_name="mediapipe-flask-01"
image_name="tdri/mediapipe-python-flask:0_0_1"
docker run -d --gpus all --name ${container_name} \
    -p 5000:5000 \
    ${image_name} \
    tail -f /dev/null
```
### direct run flask 
```
container_name="mediapipe-flask-01"
image_name="tdri/mediapipe-python-flask:0_0_1"
docker run all -d --rm --name ${container_name} \
    -p 5000:5000 \
    ${image_name} \
    flask run --host=0.0.0.0 --no-reload

```

