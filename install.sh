apt update 

apt install git
apt install unzip

unzip fake_dataset.zip

# install pytorch
apt install python3
apt install -y python3-distutils
apt install -y curl 
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py

pip install torch torchvision


# build tracker
cd tracking 

pip install opencv-python
pip install pyyaml yacs tqdm colorama matplotlib cython tensorboardX
apt-get install python3-dev
python3 setup.py build_ext --inplace

# install detectron2 
cd ../detectron2

pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
python -m pip install -e .

# install reid module
cd ../deep-person-reid/
pip install -r requirements.txt
python setup.py develop

cd ..