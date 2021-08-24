# Code and Data for Paper "SOON: Scenario Oriented Object Navigation with Graph-based Exploration" 

## Environment Installation
Download Room-to-Room navigation data:
```
bash ./tasks/R2R/data/download.sh
```

Download image features for environments:
```
mkdir img_features
wget https://www.dropbox.com/s/715bbj8yjz32ekf/ResNet-152-imagenet.zip -P img_features/
cd img_features
unzip ResNet-152-imagenet.zip
```

Python requirements: Need python3.6 (python 3.5 should be OK since I removed the allennlp dependencies)
```
pip install -r python_requirements.txt
```

Install Matterport3D simulators:
```
git submodule update --init --recursive 
sudo apt-get install libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev
mkdir build && cd build
cmake -DEGL_RENDERING=ON ..
make -j8
```

## Code

### Speaker Training
```
bash run/speaker.bash 0
```
0 is the id of GPU. It will train the speaker and save the snapshot under snap/speaker/

### Agent Training
```
python3.6 -u r2r_src/train.py --upload --dataset SOON --maxAction 20 --log_every 500 --rl_ml_weight 1 --det_loss
```

### Evaluation

```
sh run/eval.sh
```

