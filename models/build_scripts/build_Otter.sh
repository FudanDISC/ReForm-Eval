currentPath=$(dirname "$0")
cd $currentPath/../Otter
conda env create -f environment.yml
conda activate otter
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip uninstall transformers
pip install transformers==4.29.0
pip install scikit-learn
pip install SentencePiece
