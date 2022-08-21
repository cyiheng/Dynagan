FILE=$1

echo "Note: available model is dynagan"
echo "Specified [$FILE]"

mkdir -p ./checkpoints/${FILE}_pretrained
MODEL_FILE=./checkpoints/${FILE}_pretrained/latest_net_G.pth
URL=https://ubocloud.univ-brest.fr/s/fT6saSzZjGCbmAS/download/$FILE.pth

wget -N $URL -O $MODEL_FILE
