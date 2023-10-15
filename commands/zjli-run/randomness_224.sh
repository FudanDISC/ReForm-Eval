conda activate blip2
python utils/randomess_runner.py --models blip2_t5,instructblip_t5,instructblip_vicuna  --devices 0,1,2,3,4,5,6,7
conda activate llmv-bench
python utils/randomess_runner.py --models llava_v0,llava_llama2  --devices 0,1,2,3,4,5,6,7
conda activate minigpt4
python utils/randomess_runner.py --models minigpt4  --devices 0,1,2,3,4,5,6,7
conda activate mplug_owl
python utils/randomess_runner.py --models mplug_owl  --devices 0,1,2,3,4,5,6,7
conda activate llama_adapter_v2
python utils/randomess_runner.py --models llama_adapter_v2  --devices 0,1,2,3,4,5,6,7
conda activate pandagpt
python utils/randomess_runner.py --models pandagpt  --devices 0,1,2,3,4,5,6,7
conda activate lynx
python utils/randomess_runner.py --models lynx  --devices 0,1,2,3,4,5,6,7




