import json
from transformers import AutoTokenizer, AutoConfig, TextIteratorStreamer
import torch
import math
from threading import Thread

import sys
sys.path.append("../../")
from component.utils import ModelUtils


def main():
    context_size = 12288  # # #
    # 使用合并后的模型进行推理
    # model_name_or_path = '/root/autodl-tmp/output_merge/merge_llama2'  # # #
    # adapter_name_or_path = None

    # 使用base model和adapter进行推理
    model_name_or_path = '/path/to/models/Llama-2-7b-chat-hf'
    adapter_name_or_path = '/path/to/output/LongQLoRA-llama2-7b-chat-12k/checkpoint'

    # 加载数据
    with open("/path/to/test_data/paper_review_data_longqlora_infer.jsonl", 'r', encoding='utf8') as f:
        data_list = f.readlines()
    data_example = data_list[0]
    data_example = json.loads(data_example)
    ip = data_example["input"].strip()
    # op = data_example["output"].strip()
    del data_list

    # template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nUSER: {input}\nASSISTANT: "
    template = ("Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{{instruction}}\n\n### Input:\n{{input}}\n\n### Response:")  # # #
    instruction = "You are a professional machine learning conference reviewer who reviews a given paper and considers 4 criteria: ** importance and novelty **, ** potential reasons for acceptance **, ** potential reasons for rejection **, and ** suggestions for improvement **. The given paper is as follows.\n\n\n".strip()
    template = template.replace("{{instruction}}", instruction)
    


    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False
    # 生成超参配置
    gen_kwargs = {
        'max_new_tokens': 900,
        'top_p': 0.9,
        'temperature': 0.35,
        'repetition_penalty': 1.0,
        'do_sample': True
    }

    # Set RoPE scaling factor
    config = AutoConfig.from_pretrained(model_name_or_path)
    orig_ctx_len = getattr(config, "max_position_embeddings", None)  # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    # 加载模型
    model = ModelUtils.load_model(
        model_name_or_path,
        config=config,
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=adapter_name_or_path
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60.0)

    gen_kwargs['eos_token_id'] = tokenizer.eos_token_id
    gen_kwargs["streamer"] = streamer

    text = ip
    while True:
        text = text.strip()
        text = template.replace("{{input}}", text)

        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        gen_kwargs["input_ids"] = input_ids
        with torch.no_grad():
            thread = Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()
            print('Output:')

            response = []
            for new_text in streamer:
                print(new_text, end='', flush=True)
                response.append(new_text)
        print()
        text = input('Input：')


if __name__ == '__main__':
    main()
