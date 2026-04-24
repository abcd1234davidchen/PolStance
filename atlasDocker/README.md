dockerfile->image
```bash
docker build -t polstance-atlas .
```

put token in env variable
```bash
export export HF_TOKEN="THE_HF_TOKEN"
```

image->container
```bash
docker run -it -p 7860:7860 -e HF_TOKEN="$HF_TOKEN" polstance-atlas
```