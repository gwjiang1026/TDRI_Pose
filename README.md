
# How to build image
```
## build docker image
image_name="venraas/hello-python-flask-uwsgi:demo"
docker build -t "$image_name" .
```
# Run Container 
### run as deamon 
```
container_name="hello-flask-01"
image_name="venraas/hello-python-flask-uwsgi:demo"
docker run -d --name ${container_name} \
    -p 5000:5000 \
    ${image_name} \
    tail -f /dev/null
```
### direct run flask 
```
container_name="hello-flask-01"
image_name="venraas/hello-python-flask-uwsgi:demo"
docker run -d --rm --name ${container_name} \
    -p 5000:5000 \
    ${image_name} \
    flask run --host=0.0.0.0 --no-reload

```

