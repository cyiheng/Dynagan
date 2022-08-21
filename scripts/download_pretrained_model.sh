echo "Download pretrained model..."

mkdir -p ./checkpoints/pretrained_model
MODEL_FILE=./checkpoints/pretrained_model/latest_net_G.pth
URL=https://ubocloud.univ-brest.fr/s/fT6saSzZjGCbmAS/download/pretrained_model.pth

wget -N $URL -O $MODEL_FILE
