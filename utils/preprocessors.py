"""
a preprocessor to encode the conversation-like inputs
references: https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
"""
import random
from typing import Any
from copy import deepcopy
from PIL import Image
alphabet = ['abcdefghijklmnopqrstuvwxyz',
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            '123456789']

class BaseProcessor(object):
    def __init__(self, sep, sep2, roles=['Question', 'Answer'], alphabet_choice=None, infer_method='generation'):
        self.sep = sep
        self.sep2 = sep2
        self.roles = roles
        if alphabet_choice is not None:
            if alphabet_choice == 'number':
                self.ab = alphabet[2]
            elif alphabet_choice == 'lower':
                self.ab = alphabet[0]
            elif alphabet_choice == 'upper':
                self.ab = alphabet[1]
            else:
                raise ValueError
        else:
            self.ab = alphabet

    def preprocess(self, item):
        """
        A function to gather structural information into a text prompt
        Parameters:
            item: {
                "instruct": the main instruct for the task
                "question": the main question input
                "answer_options": a list of candidate options
                "history": a list of the dialog history [{"from": "human", "value": "xxx"}, {"from": "assistant", "value": "yyy"}]
            }
        Returns:
            ret: the constructed prompt
        """
        if 'instruct' in item:
            instruct = item['instruct'] + self.sep
        else:
            instruct = ''

        ret = instruct
        
        seps = [self.sep, self.sep2]
        if 'history' in item:
            for i, res in enumerate(item['history']):
                ret += self.roles[i % 2] + ': ' + res['value'] + seps[i % 2]
        
        if 'question' in item:
            ret += self.roles[0] + ': ' + item['question'] + seps[0]
        
        return ret + self.roles[1] + ':'
    
    def __call__(self, item):
        return self.preprocess(item)
    
class SingleChoiceProcessor(object):
    def __init__(self, sep, sep2=None, roles=['Question', 'Answer'], alphabet_choice=None, infer_method='generation'):
        self.sep = sep
        self.sep2 = sep2
        self.roles = roles
        if alphabet_choice is not None:
            if alphabet_choice == 'number':
                self.ab = alphabet[2]
            elif alphabet_choice == 'lower':
                self.ab = alphabet[0]
            elif alphabet_choice == 'upper':
                self.ab = alphabet[1]
            else:
                raise ValueError
        else:
            self.ab = alphabet
        self.infer_method = infer_method
    
    def set_mark(self, mark_choice=None):
        if mark_choice is not None:
            if mark_choice == 'number':
                self.ab = alphabet[2]
            elif mark_choice == 'lower':
                self.ab = alphabet[0]
            elif mark_choice == 'upper':
                self.ab = alphabet[1]
            elif mark_choice == 'random':
                self.ab = alphabet
            else:
                raise ValueError
        else:
            self.ab = alphabet
    
    
    def preprocess(self, item):
        if 'instruct' in item:
            instruct = item['instruct'] + '\n' # self.sep
        else:
            instruct = ''

        ret = instruct
        
        seps = [self.sep, self.sep2]
        if 'history' in item:
            for i, res in enumerate(item['history']):
                ret += self.roles[i % 2] + ': ' + res['value'] + seps[i % 2]
        
        if 'question' in item:
            ret += self.roles[0] + ': ' + item['question'] # + seps[0]
        
        if 'answer_options' in item and self.infer_method != 'likelihood':
            ret += ' Options: '
            if isinstance(self.ab, list):
                current_ab = random.choice(self.ab)
            else:
                current_ab = self.ab
            for i, opt in enumerate(item['answer_options']):
                ret += '({}) {}'.format(current_ab[i], opt)
                if i == len(item['answer_options']) - 1:
                    ret += '.' # + seps[0]
                else:
                    ret += '; '
        
        # if 'answer_options' in item and self.infer_method == 'likelihood':
        #     ret += ' Options: '
        #     if isinstance(self.ab, list):
        #         current_ab = random.choice(self.ab)
        #     else:
        #         current_ab = self.ab
        #     for i, opt in enumerate(item['answer_options']):
        #         ret += opt
        #         if i == len(item['answer_options']) - 1:
        #             ret += '.' # + seps[0]
        #         else:
        #             ret += '; '
        return ret + seps[0] + self.roles[1] + ':'
    
    def process_qa(self, question, options=None, answer=None):
        full_question = question
        if options is not None:
            full_question += ' Options: '
            if isinstance(self.ab, list):
                current_ab = random.choice(self.ab)
            else:
                current_ab = self.ab
            for i, opt in enumerate(options):
                full_question += '({}) {}'.format(current_ab[i], opt)
                if i == len(options) - 1:
                    full_question += '.' # + seps[0]
                else:
                    full_question += '; '
        if type(answer) == int:
            full_answer = '({}) {}'.format(current_ab[answer], options[answer])
        else:
            if answer in options:
                index = options.index(answer)
                full_answer = '({}) {}'.format(current_ab[index], answer)
            else:
                full_answer = answer
        return full_question, full_answer
    
    def __call__(self, item):
        return self.preprocess(item)
    
class ConvSingleChoiceProcessor(object):
    def __init__(self, sep, sep2=None, roles=['Question', 'Answer'], system_msg=None, first_query_fn=None, \
                 init_conv=None, sep_style='two', alphabet_choice=None, infer_method='generation', response_prefix=None):
        self.sep = sep
        self.sep2 = sep2
        self.roles = roles
        self.roles_map = {'human': roles[0], 'assistant': roles[1]}
        self.system_msg = system_msg
        self.first_query_proc_fn = first_query_fn
        self.init_conv = init_conv
        self.sep_style = sep_style
        if alphabet_choice is not None:
            if alphabet_choice == 'number':
                self.ab = alphabet[2]
            elif alphabet_choice == 'lower':
                self.ab = alphabet[0]
            elif alphabet_choice == 'upper':
                self.ab = alphabet[1]
            else:
                raise ValueError
        else:
            self.ab = alphabet
        self.infer_method = infer_method
        self.response_prefix = None if infer_method=='likelihood' else response_prefix
    
    def set_mark(self, mark_choice=None):
        if mark_choice is not None:
            if mark_choice == 'number':
                self.ab = alphabet[2]
            elif mark_choice == 'lower':
                self.ab = alphabet[0]
            elif mark_choice == 'upper':
                self.ab = alphabet[1]
            elif mark_choice == 'random':
                self.ab = alphabet
            else:
                raise ValueError
        else:
            self.ab = alphabet
    
    
    def preprocess(self, item):
        current_conv = []
            
        if self.init_conv is not None:
            current_conv.extend([[msg[0], msg[1]] for msg in self.init_conv])
            offset = len(self.init_conv)
        else:
            offset = 0
        first_query = ''
        if 'instruct' in item and self.sep_style != 'llama_adapter2':
            instruct = item['instruct'] + ' '
        else:
            instruct = ''

        first_query = first_query + instruct

        # process the main target question / query
        main_query = self.process_main_query(item)

        if 'history' in item:
            # the history should be the first query
            if len(item['history']) > 0:
                first_query = first_query + item['history'][0]['value']
                current_conv.append([self.roles_map['human'], first_query])
                current_conv.extend([[self.roles_map[msg['from']], msg['value']] for msg in item['history'][1:]])
                current_conv.append([self.roles_map['human'], main_query])
            else:
                # if no history is provided, using the main question
                current_conv.append([self.roles_map['human'], first_query + main_query])
        else:
            # if no history is provided, using the main question
            current_conv.append([self.roles_map['human'], first_query + main_query])

        # append the conversation for response
        current_conv.append([self.roles_map['assistant'], ''])
        # if self.response_prefix is None:
        #     current_conv.append([self.roles_map['assistant'], ''])
        # else:
        #     current_conv.append([self.roles_map['assistant'], self.response_prefix])

        # process the first query as required in LLaVA
        if self.first_query_proc_fn is not None:
            current_conv[offset][1] = self.first_query_proc_fn(current_conv[offset][1])
        
        # make the conversation
        full_conversation = ''
        # print(current_conv)
        if self.sep_style == 'two':
            seps = [self.sep, self.sep2]
            if self.system_msg is not None:
                full_conversation += self.system_msg + seps[0]
            for i, (role, msg) in enumerate(current_conv):
                if msg:
                    full_conversation += role + ': ' + msg + seps[i%2]
                else:
                    if self.response_prefix is not None:
                        full_conversation += role + ': ' + self.response_prefix
                    else:
                        full_conversation += role + ':'
        elif self.sep_style == 'one':
            sep = self.sep
            if self.system_msg is not None:
                full_conversation += self.system_msg + sep
            for i, (role, msg) in enumerate(current_conv):
                if msg:
                    full_conversation += role + ': ' + msg + sep
                else:
                    if self.response_prefix is not None:
                        full_conversation += role + ': ' + self.response_prefix
                    else:
                        full_conversation += role + ':'
        elif self.sep_style == 'llama_2':
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            full_conversation = ""
            for i, (role, message) in enumerate(current_conv):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system_msg) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        full_conversation += self.sep + message
                    else:
                        full_conversation += " " + message + " " + self.sep2
                else:
                    if self.response_prefix is not None:
                        full_conversation += self.response_prefix
                    else:
                        full_conversation += ''
                    full_conversation += ""
            full_conversation = full_conversation.lstrip(self.sep)
        elif self.sep_style == 'otter':
            seps = [self.sep, self.sep2]
            if self.system_msg is not None:
                full_conversation += self.system_msg
            for i, (role, msg) in enumerate(current_conv):
                if msg:
                    if i%2==0:
                        full_conversation += role + ': ' + msg + seps[0]
                    else:
                        full_conversation += role + ':' + '<answer> ' + msg + seps[1]
                else:
                    if self.response_prefix is not None:
                        full_conversation += role + ':' + '<answer> ' + self.response_prefix
                    else:
                        full_conversation += role + ':' + '<answer>'
                    
        elif self.sep_style == 'llama_adapter2':
            sep = self.sep
            if self.system_msg is not None:
                full_conversation += self.system_msg + sep
            # add the instruction
            if 'instruct' in item:
                full_conversation += 'Instruction:\n{}'.format(item['instruct']) + sep
            for i, (role, msg) in enumerate(current_conv):
                if msg:
                    full_conversation += role + ':\n' + msg + sep
                else:
                    if self.response_prefix is not None:
                        full_conversation += role + ':\n' + self.response_prefix
                    else:
                        full_conversation += role + ':'
        # elif self.sep_style == 'lavin':
        #     sep = self.sep
        #     if 'instruct' in item:
        #         full_conversation += 'Context:\n{}'.format(item['instruct']) + sep
        #     for i, (role, msg) in enumerate(current_conv):
        #         if msg:
        #             full_conversation += role + ': ' + msg + sep
        #         else:
        #             full_conversation += role + ': '
        elif self.sep_style == 'visualglm':
            sep = self.sep
            if self.system_msg is not None:
                full_conversation += self.system_msg + sep
            for i, (role, msg) in enumerate(current_conv):
                if msg:
                    if i%2==0:
                        full_conversation += '[Round {}]\n'.format(int(i/2+1)) + role + '：' + msg + sep
                    else:
                        full_conversation += role + '：' + msg + sep
                else:
                    if self.response_prefix is not None:
                        full_conversation += role + '：' + self.response_prefix
                    else:
                        full_conversation += role + '：'

        else:
            raise NotImplementedError
        
        return full_conversation
    
    def process_main_query(self, item):
        ret = ''
        if 'question' in item:
            ret += item['question']

        if 'answer_options' in item and self.infer_method != 'likelihood':
            ret += ' Options: '
            if isinstance(self.ab, list):
                current_ab = random.choice(self.ab)
            else:
                current_ab = self.ab
            for i, opt in enumerate(item['answer_options']):
                ret += '({}) {}'.format(current_ab[i], opt)
                if i == len(item['answer_options']) - 1:
                    ret += '.' # + seps[0]
                else:
                    ret += '; '
        return ret
    
    def process_qa(self, question, options=None, answer=None):
        full_question = question
        if options is not None:
            full_question += ' Options: '
            if isinstance(self.ab, list):
                current_ab = random.choice(self.ab)
            else:
                current_ab = self.ab
            for i, opt in enumerate(options):
                full_question += '({}) {}'.format(current_ab[i], opt)
                if i == len(options) - 1:
                    full_question += '.' # + seps[0]
                else:
                    full_question += '; '
        if type(answer) == int:
            full_answer = '({}) {}'.format(current_ab[answer], options[answer])
        else:
            if answer in options:
                index = options.index(answer)
                full_answer = '({}) {}'.format(current_ab[index], answer)
            else:
                full_answer = answer
        if self.response_prefix is not None:
            full_answer = self.response_prefix + ' ' + full_answer
        return full_question, full_answer


    
    def __call__(self, item):
        return self.preprocess(item)

TEMPLATE = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

class MMGPTSingleChoiceProcessor(object):
    def __init__(self, sep, sep2=None, roles=['Instruction', 'Response'], first_query_fn=None, init_conv=None, sep_style='two', \
                 alphabet_choice=None, infer_method='generation', response_prefix=None):
        self.sep = sep
        self.sep2 = sep2
        self.roles = roles
        self.system_msg = TEMPLATE
        self.first_query_proc_fn = first_query_fn
        self.init_conv = init_conv
        self.sep_style = sep_style
        if alphabet_choice is not None:
            if alphabet_choice == 'number':
                self.ab = alphabet[2]
            elif alphabet_choice == 'lower':
                self.ab = alphabet[0]
            elif alphabet_choice == 'upper':
                self.ab = alphabet[1]
            else:
                raise ValueError
        else:
            self.ab = alphabet
        self.infer_method = infer_method
        self.response_prefix = None if infer_method=='likelihood' else response_prefix
    
    def set_mark(self, mark_choice=None):
        if mark_choice is not None:
            if mark_choice == 'number':
                self.ab = alphabet[2]
            elif mark_choice == 'lower':
                self.ab = alphabet[0]
            elif mark_choice == 'upper':
                self.ab = alphabet[1]
            elif mark_choice == 'random':
                self.ab = alphabet
            else:
                raise ValueError
        else:
            self.ab = alphabet
    
    
    def preprocess(self, item):
        current_conv = []
        if self.init_conv is not None:
            current_conv.extend([msg[1] for msg in self.init_conv])
            offset = len(self.init_conv)
        else:
            offset = 0
        first_query = ''
        if 'instruct' in item:
            instruct = item['instruct'] + ' '
        else:
            instruct = ''

        first_query = first_query + instruct

        # process the main target question / query
        main_query = self.process_main_query(item)

        if 'history' in item:
            # the history should be the first query
            if len(item['history']) > 0:
                first_query = first_query + item['history'][0]['value']
                current_conv.append(first_query)
                current_conv.extend([msg['value'] for msg in item['history'][1:]])
                current_conv.append(main_query)
            else:
                # if no history is provided, using the main question
                current_conv.append(first_query + main_query)
        else:
            # if no history is provided, using the main question
            current_conv.append(first_query + main_query)

        # append the conversation for response
        current_conv.append('')
        
        # process the first query as required in LLaVA
        if self.first_query_proc_fn is not None:
            current_conv[offset] = self.first_query_proc_fn(current_conv[offset])
        
        # make the conversation
        full_conversation = ''
        if self.sep_style == 'two':
            seps = [self.sep, self.sep2]
            if self.system_msg is not None:
                full_conversation += self.system_msg + seps[0]
            full_conversation += "Image:\n<image>" + seps[1]
            for i, msg in enumerate(current_conv):
                if msg:
                    full_conversation += self.roles[i%2] + ':\n' + msg + seps[i%2]
                else:
                    if self.response_prefix is None:
                        full_conversation += self.roles[i%2] + ':\n'
                    else:
                        full_conversation += self.roles[i%2] + ':\n' + self.response_prefix
        elif self.sep_style == 'one':
            sep = self.sep
            if self.system_msg is not None:
                full_conversation += self.system_msg + sep
            full_conversation += "Image:\n<image>" + sep
            for i, msg in enumerate(current_conv):
                if msg:
                    full_conversation += self.roles[i%2] + ':\n' + msg + sep
                else:
                    if self.response_prefix is None:
                        full_conversation += self.roles[i%2] + ':\n'
                    else:
                        full_conversation += self.roles[i%2] + ':\n' + self.response_prefix
        else:
            raise NotImplementedError
               
        return full_conversation
    
    def process_main_query(self, item):
        ret = ''
        if 'question' in item:
            ret += item['question']

        if 'answer_options' in item and self.infer_method != 'likelihood':
            ret += ' Options: '
            if isinstance(self.ab, list):
                current_ab = random.choice(self.ab)
            else:
                current_ab = self.ab
            for i, opt in enumerate(item['answer_options']):
                ret += '({}) {}'.format(current_ab[i], opt)
                if i == len(item['answer_options']) - 1:
                    ret += '.' # + seps[0]
                else:
                    ret += '; '
        return ret
    
    def process_qa(self, question, options=None, answer=None):
        full_question = question
        if options is not None:
            full_question += ' Options: '
            if isinstance(self.ab, list):
                current_ab = random.choice(self.ab)
            else:
                current_ab = self.ab
            for i, opt in enumerate(options):
                full_question += '({}) {}'.format(current_ab[i], opt)
                if i == len(options) - 1:
                    full_question += '.' # + seps[0]
                else:
                    full_question += '; '
        if type(answer) == int:
            full_answer = '({}) {}'.format(current_ab[answer], options[answer])
        else:
            if answer in options:
                index = options.index(answer)
                full_answer = '({}) {}'.format(current_ab[index], answer)
            else:
                full_answer = answer
        if self.response_prefix is not None:
            full_answer = self.response_prefix + ' ' + full_answer
        return full_question, full_answer


    
    def __call__(self, item):
        return self.preprocess(item)
    

# processor for shikra
class ShikraProcessor(object):
    def __init__(self, ds, alphabet_choice=None, infer_method='generation', answer_prefix=None):
        self.ds_template = ds
        self.roles_map = {'human': ds.roles[0], 'assistant': ds.roles[1]}
        if alphabet_choice is not None:
            if alphabet_choice == 'number':
                self.ab = alphabet[2]
            elif alphabet_choice == 'lower':
                self.ab = alphabet[0]
            elif alphabet_choice == 'upper':
                self.ab = alphabet[1]
            else:
                raise ValueError
        else:
            self.ab = alphabet
        self.infer_method = infer_method
        self.response_prefix = None if infer_method=='likelihood' else answer_prefix
    
    def set_mark(self, mark_choice=None):
        if mark_choice is not None:
            if mark_choice == 'number':
                self.ab = alphabet[2]
            elif mark_choice == 'lower':
                self.ab = alphabet[0]
            elif mark_choice == 'upper':
                self.ab = alphabet[1]
            elif mark_choice == 'random':
                self.ab = alphabet
            else:
                raise ValueError
        else:
            self.ab = alphabet
    
    
    def preprocess(self, item):
        current_conv = []
            
        first_query = ''
        if 'instruct' in item:
            instruct = item['instruct'] + ' '
        else:
            instruct = ''

        first_query = first_query + instruct

        # process the main target question / query
        main_query = self.process_main_query(item)

        if 'history' in item:
            # the history should be the first query
            if len(item['history']) > 0:
                first_query = first_query + item['history'][0]['value']
                current_conv.append([self.roles_map['human'], first_query])
                current_conv.extend([[self.roles_map[msg['from']], msg['value']] for msg in item['history'][1:]])
                current_conv.append([self.roles_map['human'], main_query])
            else:
                # if no history is provided, using the main question
                current_conv.append([self.roles_map['human'], first_query + main_query])
        else:
            # if no history is provided, using the main question
            current_conv.append([self.roles_map['human'], first_query + main_query])

        # append the conversation for response
        if self.response_prefix is None:
            current_conv.append([self.roles_map['assistant'], ''])
        else:
            current_conv.append([self.roles_map['assistant'], self.response_prefix])
        
        current_ds = deepcopy(self.ds_template)
        # need to load the image first
        image = self.load_image(item['image'])
        image = self.expand2square(image)
        current_ds.set_image(image)
        for r_role, round in current_conv:
            if self.response_prefix is None:
                current_ds.append_message(r_role, round)
            else:
                if r_role == self.roles_map['assistant']:
                    new_info = round if round.startswith(self.response_prefix) else "{} {}".format(self.response_prefix, round)
                else:
                    new_info = round
                current_ds.append_message(r_role, new_info)
        full_inputs = current_ds.to_model_input()
        full_inputs['raw_text'] = current_ds.preprocessor['text'].batch_decode(full_inputs['input_ids'])[0].replace(' <im_patch>', '').replace(' <s>', '')
        return full_inputs
    
    def load_image(self, img):
        if type(img) == str:
            # img is the image path
            image = Image.open(img)
            image = image.convert('RGB')
            return image
        else:
            return img
    
    def expand2square(self, pil_img, background_color=(255, 255, 255)):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
    
    def process_main_query(self, item):
        ret = ''
        if 'question' in item:
            ret += item['question']

        if 'answer_options' in item and self.infer_method != 'likelihood':
            ret += ' Options: '
            if isinstance(self.ab, list):
                current_ab = random.choice(self.ab)
            else:
                current_ab = self.ab
            for i, opt in enumerate(item['answer_options']):
                ret += '({}) {}'.format(current_ab[i], opt)
                if i == len(item['answer_options']) - 1:
                    ret += '.' # + seps[0]
                else:
                    ret += '; '
        return ret
    
    def process_qa(self, question, options=None, answer=None):
        full_question = question
        if options is not None:
            full_question += ' Options: '
            if isinstance(self.ab, list):
                current_ab = random.choice(self.ab)
            else:
                current_ab = self.ab
            for i, opt in enumerate(options):
                full_question += '({}) {}'.format(current_ab[i], opt)
                if i == len(options) - 1:
                    full_question += '.' # + seps[0]
                else:
                    full_question += '; '
        if type(answer) == int:
            full_answer = '({}) {}'.format(current_ab[answer], options[answer])
        else:
            if answer in options:
                index = options.index(answer)
                full_answer = '({}) {}'.format(current_ab[index], answer)
            else:
                full_answer = answer
        if self.response_prefix is not None:
            full_answer = full_answer + ' ' + full_answer
        return full_question, full_answer

    def __call__(self, item):
        return self.preprocess(item)