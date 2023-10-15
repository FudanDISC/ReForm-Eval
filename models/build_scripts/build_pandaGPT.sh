currentPath=$(dirname "$0")
cd $currentPath/../PandaGPT
conda create -n pandagpt python=3.10 -y
conda activate pandagpt

pip install -r requirements.txt
# install pytorch
pip install scikit-learn
pip install SentencePiece