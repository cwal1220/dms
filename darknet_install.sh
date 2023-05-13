# http://shuoyang1213.me/WIDERFACE/
# tran, validation, annotation 다운로드

sudo apt update
sudo apt upgrade

git clone https://github.com/AlexeyAB/darknet

sudo apt install libopencv-dev build-essential

cd darknet

sed -i 's/OPENCV=0/OPENCV=1/g' Makefile

# GPU 사용시 입력
# sed -i 's/GPU=0/GPU=1/g' Makefile
# sed -i 's/CUDNN=0/CUDNN=1/g' Makefile

make -j $(nproc)

wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

./darknet detect cfg/yolov4-tiny.cfg yolov4-tiny.weights data/dog.jpg

# WIDER Face data 학습하기

git clone https://github.com/cwal1220/dms -b train
cd dms
sudo apt install unzip
unzip WIDER_train.zip
unzip WIDET_val.zip
unzip wider_face_split.zip

# WIDER annotation을 YOLO format으로 변경하기 위한 과정
sudo apt install python3-pip
pip3 install opencv-python tqdm
python3 convert_wider_to_yolo.py

find "$(pwd)/train" -name "*.jpg" > train.txt
find "$(pwd)/val" -name "*.jpg" > val.txt

# 가중치 받기
wget https://github.com/cwal1220/dms/releases/download/release/yolov4-face-tiny.weights

cd ..

# 학습하기
./darknet detector train dms/face.data dms/yolov4-face-tiny.cfg dms/yolov4-face-tiny.weights -map

# 검출하기
./darknet detector test dms/face.data dms/yolov4-face-tiny.cfg dms/yolov4-face-tiny.weights dms/val/wider_1122.jpg
