# YOLObook

an automated pipeline for training custom YOLO object detection models using the [YouTube Bounding Box](https://research.google.com/youtube-bb/) dataset

pick object classes from a checkbox UI, run the cells, and get a trained model; the pipeline handles video downloading, frame extraction, label generation, and data splitting automatically

![linkedin-version(1)](https://github.com/c-w-a/YOLOv5-DeepLearning-Notebook/assets/108597555/0874883d-046b-489e-9ede-a67f55448546)

## how it works

the notebook (`YoloBook.ipynb`) walks through 9 steps:

1. **select classes** -> choose from 23 object categories (animals, vehicles, etc.)
2. **choose model size** -> yolo11n through yolo11x
3. **environment setup** -> auto-detects Colab vs local
4. **validate** -> checks configuration before committing to a long run
5. **generate config** -> creates dataset YAML and parameterized processing script
6. **process data** -> downloads YouTube videos, extracts frames at labeled timestamps, generates YOLO-format bounding box annotations, remaps class IDs, and splits into train/val/test
7. **train** -> runs YOLO11 training with the generated dataset
8. **inspect** -> visual QA with random bounding box overlays
9. **test** -> run inference on your own images or video

### data pipeline:

the core processing file (`src/utility/process-data.py`) converts raw YouTube Bounding Box CSV annotations into a training-ready dataset:

YouTube-BB CSVs → filter by selected classes → download videos (yt-dlp, 7 threads) → extract frames at annotated timestamps (ffmpeg, 10 threads) → generate YOLO-format labels → remap class IDs to zero-indexed → split into train/val/test → cleanup

each step runs with multithreaded workers where possible!

## setup

### Google Colab (most simple)

upload the repo to Google Drive, open `YoloBook.ipynb` in Colab, and run it; dependencies are installed automatically and data is stored on your Google Drive

### local

requires Python 3.8+, a CUDA-capable GPU, and ffmpeg installed on your system

```bash
git clone https://github.com/c-w-a/yolobook.git
cd yolobook
pip install -r requirements.txt
```

you also need the YouTube Bounding Box CSV files (`yt_bb_detection_train.csv` and `yt_bb_detection_validation.csv`) placed in `data/raw/`; these can be downloaded from the [YouTube-BB dataset page](https://research.google.com/youtube-bb/).

then open the notebook:
```bash
jupyter notebook YoloBook.ipynb
```

## supported classes

person, cat, dog, horse, cow, elephant, bear, zebra, giraffe, car, truck, bus, boat, motorcycle, airplane, train, bicycle, potted plant, toilet, skateboard, knife, umbrella

these are the 23 classes from the YouTube-BB dataset that overlap with COCO, enabling transfer learning from pre-trained YOLO11 weights

## license

AGPL-3.0
