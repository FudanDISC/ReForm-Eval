currentPath=$(dirname "$0")
cd $currentPath/../Multimodal-GPT
pip install -r requirements.txt
pip install -v -e .
pip install transformers==4.31