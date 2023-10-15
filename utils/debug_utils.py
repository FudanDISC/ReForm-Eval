import random
from .preprocessors import SingleChoiceProcessor, ConvSingleChoiceProcessor, ShikraProcessor, MMGPTSingleChoiceProcessor
from build import build_dataset
from utils.run_utils import *
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, SequentialSampler

def datasetdebug(args):
    # mplug-owl
    # preprocessor = ConvSingleChoiceProcessor(sep='\n', infer_method=args.infer_method, system_msg="The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.", \
    #                                          init_conv=[['Human', "<image>"]], roles=['Human', 'AI'], sep_style="one")
    # llava v0
    # preprocessor = ConvSingleChoiceProcessor("###", sep2=None, roles=('Human', 'Assistant'), system_msg="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.", \
    #                                  first_query_fn=lambda qs: "<image>\n"+qs, init_conv=[("Human", "Hi!"), ("Assistant", "Hi there! How can I help you today?")], \
    #                                  sep_style="one", infer_method="generation", response_prefix='The answer is')
    # minigpt4
    # preprocessor = ConvSingleChoiceProcessor("###", sep2=None, roles=('Human', 'Assistant'), system_msg="Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.", \
    #                                  first_query_fn=lambda qs: "<Img><ImageHere></Img> "+qs, init_conv=[], \
    #                                  sep_style="one", infer_method="generation")
    # llava-llama-2
    # preprocessor = ConvSingleChoiceProcessor("<s>", sep2="</s>", roles=('USER', 'ASSISTANT'), system_msg="You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.", \
    #                                  first_query_fn=lambda qs: "<image>\n"+qs, init_conv=[], \
    #                                  sep_style="llama_2", infer_method="generation")
    # Otter
    # preprocessor = ConvSingleChoiceProcessor(sep=' ', sep2='<|endofchunk|>', infer_method=args.infer_method, system_msg="<image>", \
    #                                          roles=['User', 'GPT'], sep_style="two")
    # minigpt4
    # preprocessor = ConvSingleChoiceProcessor("\n", sep2=None, roles=('User', 'Bot'), init_conv=[], \
    #                                  sep_style="one", infer_method="generation", response_prefix='The answer is')
    # llama adapter v1
    # preprocessor = ConvSingleChoiceProcessor(sep='\n\n### ', infer_method='generation',\
    #                        system_msg='Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.', \
    #                        roles=['Input', 'Response'], sep_style="llama_adapter2")
    # preprocessor = ConvSingleChoiceProcessor(sep='\n\n### ', infer_method='inference_method',\
    #                        system_msg='Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.', \
    #                        roles=['Input', 'Response'], sep_style="llama_adapter2")
    # pandagp4-4
    # preprocessor = ConvSingleChoiceProcessor(sep='\n', sep2='\n', infer_method='generation',\
    #                                          first_query_fn=lambda x: "<Img></Img> " + x, \
    #                        roles=['### Human ', '### Assistant'], sep_style="two")  
    # lavin
    # preprocessor = ConvSingleChoiceProcessor(sep='\n', infer_method='generation',\
    #                     roles=['Question', 'Response'], sep_style="one")
    # shikra
    # preprocessor = ShikraProcessor(ds_template, infer_method=model_args['inference_method'], answer_prefix='The answer is')
    # mmgpt
    preprocessor = MMGPTSingleChoiceProcessor("\n\n### ", roles=["Instruction", "Response"], \
                                     sep_style="one", infer_method='generation', response_prefix='The answer is')
    
    preprocessor.set_mark(args.option_mark)
    ds = build_dataset(args, args.dataset_name, args.formulation, args.dataset_config, 
                                 preprocessor)
    random_index = random.randint(0, len(ds))
    sampler = SequentialSampler(ds) if not args.distributed else DistributedSampler(ds, shuffle=False)
    dataloader = DataLoader(ds, num_workers=args.num_workers, sampler=sampler, batch_size=args.per_gpu_eval_batch_size, collate_fn=naive_list_collate_fn)
    if args.local_rank <= 0:
        print('batch:', next(iter(dataloader)))
        print('examples in the dataset:')
        print('{}-th sample:'.format(random_index+1), ds[random_index+1])
        print('{}-th sample:'.format(random_index), ds[random_index])
        print('{}-th sample:'.format(random_index-1), ds[random_index-1])