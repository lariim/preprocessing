# preprocessing

git clone https://github.com/lariim/preprocessing.git

conda env create -f environment.yml

conda activate preprocess

pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose

mkdir -p checkpoints

Models to put in checkpoints \n
https://drive.google.com/drive/folders/14wRN7RV-p4Qqh215hE9gGGJExsSJEFOn?usp=drive_link

Image in Input \n
mkdir -p Input

mkdir -p Output/{agnostic,agnostic_bottom,dense,parse,pose/{img,json}}
