# preprocessing

conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

conda install cuda-libraries-static=12.1 cuda-libraries=12.1 cuda-compiler=12.1 cuda-runtime=12.1  cuda-libraries-dev=12.1

pip install tqdm ninja opencv-python

pip install matplotlib

pip install scipy

pip install scikit-image

python -m pip install -e detectron2

sudo apt-get install ffmpeg

pip install av

mkdir -p Input

mkdir -p Output/{agnostic,agnostic_bottom,dense,parse,pose/{img,json}}
