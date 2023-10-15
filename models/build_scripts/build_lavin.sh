currentPath=$(dirname "$0")
cd $currentPath/../LaVIN
conda create -n lavin python=3.8 -y
conda activate lavin

# install pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch

# install dependency and lavin

pip install -e .
pip install scikit-learn