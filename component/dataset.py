from loguru import logger
import json
from torch.utils.data import Dataset
import numpy as np


class PretrainDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length, ignore_index=-100):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()

        logger.info("there are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        text = json.loads(data)['text']
        return text


class EvalDataset(Dataset):
    """
    用于评测ppl
    """
    def __init__(self, file, tokenizer, max_seq_length, ignore_index=-100, sliding_window=256):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.pad_token_id = tokenizer.pad_token_id
        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        token_list = np.memmap(file, dtype=np.uint16, mode='r').tolist()

        # 以滑动窗口截取评测数据
        eval_data_list = []
        for i in range(0, len(token_list), sliding_window):
            input_ids = token_list[i: i+max_seq_length]
            labels = token_list[i: i+max_seq_length]
            # padding
            padding_len = self.max_seq_length - len(input_ids)
            input_ids += [self.pad_token_id]*padding_len
            labels += [self.ignore_index]*padding_len
            eval_data_list.append({
                'input_ids': input_ids,
                'labels': labels
            })
        logger.info("there are {} data in eval dataset".format(len(eval_data_list)))
        self.data_list = eval_data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        return data


class VicunaSFTDataset(Dataset):

    def __init__(self, file, tokenizer, max_seq_length, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()

        logger.info("there are {} data in dataset".format(len(data_list)))
        self.data_list = data_list
        self.input_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nUSER: {input}\nASSISTANT: "

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """
        沿袭Vicuna的的格式。
        A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
        USER: xxx
        ASSISTANT: xxx
        """
        data = self.data_list[index]
        data = json.loads(data)
        inputs = data['input'].strip()
        output = data['output'].strip()
        # 输入部分
        input_format = self.input_template.format(input=inputs)

        input_format_ids = self.tokenizer(input_format, add_special_tokens=False).input_ids
        output_ids = self.tokenizer(output, add_special_tokens=False).input_ids + [self.eos_token_id]

        input_ids = input_format_ids + output_ids
        labels = [self.ignore_index] * len(input_format_ids) + output_ids
        assert len(input_ids) == len(labels)

        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        labels = labels[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        # padding
        padding_len = self.max_seq_length - len(input_ids)
        input_ids += [self.pad_token_id] * padding_len
        labels += [self.ignore_index] * padding_len
        attention_mask += [0] * padding_len

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        return inputs


class Llama2SFTDataset(Dataset):

    def __init__(self, file, tokenizer, max_seq_length, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()

        logger.info("there are {} data in dataset".format(len(data_list)))
        #self.data_list = [da for (di, da) in enumerate(data_list) if di not in [4356]]
        self.data_list = data_list
        logger.info("there are {} data in dataset".format(len(self.data_list)))
        #self.input_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nUSER: {input}\nASSISTANT: "
        self.input_template = ("Below is an instruction that describes a task, paired with an input that provides further context. "
                                "Write a response that appropriately completes the request.\n\n"
                                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:")
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """
        沿袭Vicuna的的格式。
        A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
        USER: xxx
        ASSISTANT: xxx
        """
        data = self.data_list[index]
        data = json.loads(data)
        inputs = data['input'].strip()
        output = data['output'].strip()
        # 输入部分
        #instruction = "You are a professional machine learning conference reviewer who reviews a given paper and considers 4 criteri"
        instruction ="You are a professional machine learning conference reviewer who reviews a given paper and considers 4 criteria: ** importance and novelty **, ** potential reasons for acceptance **, ** potential reasons for rejection **, and ** suggestions for improvement **. The given paper is as follows.:"
        instruction = instruction.strip()
        input_format = self.input_template.format(instruction=instruction, input=inputs)
        #input_format = self.input_template.format(input=inputs)

        # source_ids
        input_format_ids = self.tokenizer.encode(self.tokenizer.bos_token + input_format, add_special_tokens=False)
        # target_ids
        output_ids = self.tokenizer.encode(output + self.tokenizer.eos_token, add_special_tokens=False)

        # 分别计算输入、输出进行截断
        max_output_len = int(self.max_seq_length * (len(output_ids) / (len(input_format_ids) + len(output_ids))))
        max_output_len = max(max_output_len, 1)     #至少保留1个token的output
        max_input_len = self.max_seq_length - max_output_len

        #对输入、输出进行截断
        if len(input_format_ids) > max_input_len:
            input_format_ids = input_format_ids[:max_input_len]
        if len(output_ids) > max_output_len:
            output_ids = output_ids[:max_output_len]

        #mask
        input_format_mask = [self.ignore_index] * len(input_format_ids)

        # concat inputs
        input_ids = input_format_ids + output_ids
        labels = input_format_mask + output_ids
        

        # 再次检查截断
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]


        attention_mask = [1] * len(input_ids)

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        return inputs