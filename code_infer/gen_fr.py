"""

This short_code_main is used for testing recent foundation models
"""
# from datetime import time
import time

import torch

from prompt import TruthfulQA_generation_prompt, MGSM_generation_prompt
from load_data_02 import load_TruthfulQA, load_MGSM

global_devices = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]
max_memory = {k: '40GB' for k in global_devices}


def llama_hugging(data, model_name='llama-7b', gen_prompt=TruthfulQA_generation_prompt, w_file=None,
                  temperature=None, max_new_tokens=None, top_k=None, k=1):
    from transformers.models.llama import LlamaTokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    llama_model = AutoModelForCausalLM.from_pretrained('./foundation_models/{}'.format(model_name),
                                                   low_cpu_mem_usage=True, device_map='balanced',
                                                   torch_dtype=torch.float32, max_memory=max_memory)
    tokenizer = LlamaTokenizer.from_pretrained('./foundation_models/{}'.format(model_name))

    if model_name not in data.columns:
        data[model_name] = ''
        data[model_name].fillna('', inplace=True)
        data[model_name] = data[model_name].astype(str)

    for index, row in data.iterrows():
        # if index < 40:
        #     continue
        question = row['Question']
        print(question)
        prompt = gen_prompt.format(question)
        inputs = tokenizer(prompt, return_tensors="pt")
        k_answers = []
        count = 0
        start = time.time()
        while count < k:
            generate_ids = llama_model.generate(inputs.input_ids.to(llama_model.device),
                                                # max_new_tokens=500, top_k=40, top_p=0.9, do_sample=False, temperature=0.8, early_stopping=True,
                                                max_new_tokens=100, top_k=10, top_p=0.9, do_sample=True, temperature=0.8, early_stopping=True,
                                                use_cache=True
                                                )
            answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            answer_new = answer.split('La réponse est 29.')[-1].split('\n')
            for a in answer_new:
                if a.strip().startswith('A:'): # and '答案是' in a:
                    answer = a.strip()[2:]
                    k_answers.append(answer)
                    print('>>', answer)
                    count += 1
                    break
        print('=================', time.time() - start, llama_model.device, model_name, w_file.split('/')[-1])
        data.loc[index, model_name] = str(k_answers)
        if (index + 1) % 2 == 0:
            data.to_csv(w_file, index=False, encoding='utf-8')
    data.to_csv(w_file, index=False, encoding='utf-8')


def alpaca(data, model_name='alpaca-7b', gen_prompt=None, w_file=None, temperature=None, max_new_tokens=None, top_k=None):
    # Convert Meta's released weights into huggingface format. Follow this guide:
    # https://huggingface.co/docs/transformers/main/model_doc/llama
    # Make sure you cloned the released weight diff into your local machine. The weight diff is located at:
    # https://huggingface.co/tatsu-lab/alpaca-7b/tree/main
    # Run this function with the correct paths. E.g.,
    # python weight_diff.py recover --path_raw /home/fqt170/qinghua/project_lists/gpt_order/foundation_models/llama-7b --path_diff /home/fqt170/qinghua/project_lists/gpt_order/foundation_models/alpaca-7b-wdiff  --path_tuned /home/fqt170/qinghua/project_lists/gpt_order/foundation_models

    # weight_diff.py will be killed, so this alpaca weights is used:
    # https://huggingface.co/circulus/alpaca-7b

    # from transformers import AutoTokenizer, AutoModelForCausalLM
    # alpaca_model = AutoModelForCausalLM.from_pretrained("<path_to_store_recovered_weights>")
    # alpaca_tokenizer = AutoTokenizer.from_pretrained("<path_to_store_recovered_weights>")

    from transformers.models.llama import LlamaTokenizer, LlamaTokenizerFast, LlamaForCausalLM
    alpaca_model = LlamaForCausalLM.from_pretrained('./foundation_models/{}'.format(model_name),
                                                   low_cpu_mem_usage=True, torch_dtype=torch.float32,
                                                    max_memory=max_memory).cuda()
    alpaca_tokenizer = LlamaTokenizer.from_pretrained('./foundation_models/{}'.format(model_name))

    if model_name not in data.columns:
        data[model_name] = ''
        data[model_name].fillna('', inplace=True)
        data[model_name] = data[model_name].astype(str)

    for index, row in data.iterrows():
        question = row['Question']
        prompt = gen_prompt.format(question)
        inputs = alpaca_tokenizer(prompt, return_tensors="pt")

        # Generate
        generate_ids = alpaca_model.generate(inputs.input_ids.cuda(), max_new_tokens=max_new_tokens)
        alpaca_answer = alpaca_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        alpaca_answer = alpaca_answer.replace(gen_prompt[:-2], '')
        print(alpaca_answer)
        print('=================')
        data.loc[index, model_name] = alpaca_answer
        if (index + 1) % 20 == 0:
            data.to_csv(w_file, index=False, encoding='utf-8')
    data.to_csv(w_file, index=False, encoding='utf-8')


def openllama(data, model_name='openllama-7b-300bt', gen_prompt=None, w_file=None, temperature=None, max_new_tokens=None, top_k=None):
    # The checkpoint can be downloaded from HuggingFace Hub. https://huggingface.co/openlm-research/open_llama_7b_preview_200bt
    #  For using the weights in the transformers library, please follow the transformers LLaMA documentation.
    #  Note that we use BOS (beginning of sentence) token (id=1) during training,
    #  so it is important to prepend this token for best performance during few-shot evaluation.
    # from transformers import AutoTokenizer, LlamaForCausalLM
    # model = LlamaForCausalLM.from_pretrained('./foundations/openllama-7b-200BT')
    # tokenizer = AutoTokenizer.from_pretrained('./foundations/openllama-7b-200BT')

    from transformers.models.llama import LlamaTokenizer, LlamaTokenizerFast, LlamaForCausalLM
    openllama_model = LlamaForCausalLM.from_pretrained('./foundation_models/{}'.format(model_name),
                                                    low_cpu_mem_usage=True, torch_dtype=torch.float32,
                                                       max_memory=max_memory).cuda()
    openllama_tokenizer = LlamaTokenizer.from_pretrained('./foundation_models/{}'.format(model_name))

    if model_name not in data.columns:
        data[model_name] = ''
        data[model_name].fillna('', inplace=True)
        data[model_name] = data[model_name].astype(str)

    for index, row in data.iterrows():
        question = row['Question']
        prompt = gen_prompt.format(question)
        inputs = openllama_tokenizer(prompt, return_tensors="pt")

        # Generate
        generate_ids = openllama_model.generate(inputs.input_ids.cuda(), max_new_tokens=max_new_tokens)
        openllama_answer = openllama_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(openllama_answer)
        print('=================')
        data.loc[index, model_name] = openllama_answer
        if (index + 1) % 20 == 0:
            data.to_csv(w_file, index=False, encoding='utf-8')
    data.to_csv(w_file, index=False, encoding='utf-8')


def openalpaca(data, model_name='openalpaca_7b_preview_3bt', gen_prompt=None, w_file=None, temperature=None, max_new_tokens=None, top_k=None):
    # from transformers import LlamaForCausalLM, LlamaTokenizer
    # # the previewed version of OpenAlpaca
    # model_path = r'openllmplayground/openalpaca_7b_preview_2bt'
    # tokenizer = LlamaTokenizer.from_pretrained(model_path)
    # model = LlamaForCausalLM.from_pretrained(model_path).cuda()

    from transformers.models.llama import LlamaTokenizer, LlamaForCausalLM
    openalpaca_model = LlamaForCausalLM.from_pretrained('./foundation_models/{}'.format(model_name),
                                                        low_cpu_mem_usage=True, torch_dtype=torch.float32).cuda()
    openalpaca_tokenizer = LlamaTokenizer.from_pretrained('./foundation_models/{}'.format(model_name))

    if model_name not in data.columns:
        data[model_name] = ''
        data[model_name].fillna('', inplace=True)
        data[model_name] = data[model_name].astype(str)

    for index, row in data.iterrows():
        question = row['Question']
        prompt = gen_prompt.format(question)
        prompt_no_input = f'### Instruction:\n{prompt}\n\n### Response:'
        tokens = openalpaca_tokenizer.encode(prompt_no_input)
        # inputs = openalpaca_tokenizer(prompt, return_tensors="pt")
        bos_token_id, eos_token_id = 1, 2  # see https://github.com/openlm-research/open_llama#preview-weights-release-and-usage
        tokens = [bos_token_id] + tokens + [eos_token_id] + [bos_token_id]
        tokens = torch.LongTensor(tokens[-1024:]).unsqueeze(0).cuda()
        instance = {'input_ids': tokens,
                    'top_k': 50,
                    'top_p': 0.9,
                    'generate_len': max_new_tokens}
        length = len(tokens[0])

        # Generate
        with torch.no_grad():
            rest = openalpaca_model.generate(
                input_ids=tokens,
                max_length=length + instance['generate_len'],
                use_cache=True,
                do_sample=True,
                top_p=instance['top_p'],
                top_k=instance['top_k']
            )

        output = rest[0][length:]
        openalpaca_answer = openalpaca_tokenizer.decode(output, skip_special_tokens=False)
        openalpaca_answer = openalpaca_answer.replace('<s>', '').replace('</s>', '').strip()
        print(openalpaca_answer)
        print('=================')
        data.loc[index, model_name] = openalpaca_answer
        if (index + 1) % 20 == 0:
            data.to_csv(w_file, index=False, encoding='utf-8')
    data.to_csv(w_file, index=False, encoding='utf-8')


def redpajama(data, model_name='RedPajama-INCITE-Base-3B-v1', gen_prompt=None, w_file=None, temperature=0.7, max_new_tokens=None, top_k=50):
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM

    MIN_TRANSFORMERS_VERSION = '4.25.1'

    # check transformers version
    assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

    # init
    redpajama_model = AutoModelForCausalLM.from_pretrained("./foundation_models/{}".format(model_name),
                                                           low_cpu_mem_usage=True, torch_dtype=torch.float32).cuda()
    redpajama_model = redpajama_model.to('cuda:0')
    redpajama_tokenizer = AutoTokenizer.from_pretrained("./foundation_models/{}".format(model_name))

    if model_name not in data.columns:
        data[model_name] = ''
        data[model_name].fillna('', inplace=True)
        data[model_name] = data[model_name].astype(str)

    for index, row in data.iterrows():
        question = row['Question']
        prompt = gen_prompt.format(question)
        inputs = redpajama_tokenizer(prompt, return_tensors='pt').to(redpajama_model.device)
        input_length = inputs.input_ids.shape[1]
        outputs = redpajama_model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_p=0.7, top_k=top_k, return_dict_in_generate=True,
        )
        token = outputs.sequences[0, input_length:]
        redpajama_answer = redpajama_tokenizer.decode(token)
        print(redpajama_answer)
        print('=================')
        data.loc[index, model_name] = redpajama_answer
        if (index + 1) % 20 == 0:
            data.to_csv(w_file, index=False, encoding='utf-8')
    data.to_csv(w_file, index=False, encoding='utf-8')


def run_truthfulQA(model_name='llama-7b', altering='ex_random_two', max_new_tokens=50, evaluate=True):
    truthfuleQA_file = "../data/TruthfulQA/TruthfulQA_{}.csv".format(altering)
    truthfulQA_data = load_TruthfulQA(truthfuleQA_file)[:100]
    gen_prompt = TruthfulQA_generation_prompt
    truthfulQA_gen_file = "../data/TruthfulQA/{}_TruthfulQA_{}.csv".format(model_name, altering)

    if not evaluate:
        if model_name in ['llama-7b']:
            llama_hugging(data=truthfulQA_data, model_name=model_name, gen_prompt=gen_prompt, w_file=truthfulQA_gen_file, max_new_tokens=max_new_tokens)
        elif model_name in ['alpaca-7b']:
            alpaca(data=truthfulQA_data, model_name=model_name, gen_prompt=gen_prompt, w_file=truthfulQA_gen_file, max_new_tokens=max_new_tokens)
        elif model_name in ['openllama-7b-300bt']:
            openllama(data=truthfulQA_data, model_name=model_name, gen_prompt=gen_prompt, w_file=truthfulQA_gen_file, max_new_tokens=max_new_tokens)
        elif model_name in ['openalpaca_7b_preview_3bt']:
            openalpaca(data=truthfulQA_data, model_name=model_name, gen_prompt=gen_prompt, w_file=truthfulQA_gen_file, max_new_tokens=max_new_tokens)
        elif model_name in ['RedPajama-INCITE-Base-3B-v1']:
            redpajama(data=truthfulQA_data, model_name=model_name, gen_prompt=gen_prompt,
                      w_file=truthfulQA_gen_file, temperature=0.7, max_new_tokens=max_new_tokens)
    else:
        # evaluate
        from evaluate_05 import eval_TruthfulQA
        new_data = load_MGSM(truthfulQA_gen_file, samples_num=100)
        # eval_TruthfulQA(data=new_data, g_model=model_name, ft_model="curie:ft-personal-2023-05-08-10-18-37", w_file=truthfulQA_gen_file, metrics='GPT-judge')
        # eval_TruthfulQA(data=new_data, g_model=model_name, ft_model='curie:ft-personal-2023-05-08-10-20-38', w_file=truthfulQA_gen_file, metrics='GPT-info')
        eval_TruthfulQA(data=new_data, g_model=model_name, ft_model='', w_file=truthfulQA_gen_file, metrics='bleurt')


def run_MGSM(lan='en', model_name='llama-7b', altering='_ex_random_two', temperature=0.8, k=3, max_new_tokens=100, evaluate=True):
    mgsm_file = "./data/mgsm/mgsm_{}{}.csv".format(lan, altering)
    gen_prompt = MGSM_generation_prompt[lan]
    mgsm_data = load_MGSM(mgsm_file, samples_num=100)
    MGSM_gen_file = "./data/mgsm/{}_mgsm_{}{}.csv".format(model_name, lan, altering)

    if not evaluate:
        if model_name in ['llama-7b', 'llama-13b', 'llama-30b', 'llama-65b']:
            llama_hugging(data=mgsm_data, model_name=model_name, gen_prompt=gen_prompt, w_file=MGSM_gen_file,
                          max_new_tokens=max_new_tokens, temperature=temperature, k=k)
        elif model_name in ['alpaca-7b', 'alpaca-13b']:
            alpaca(data=mgsm_data, model_name=model_name, gen_prompt=gen_prompt, w_file=MGSM_gen_file, max_new_tokens=max_new_tokens)
        elif model_name == 'openllama-7b-300bt':
            openllama(data=mgsm_data, model_name=model_name, gen_prompt=gen_prompt, w_file=MGSM_gen_file, max_new_tokens=max_new_tokens)
        elif model_name == 'openalpaca_7b_preview_3bt':
            openalpaca(data=mgsm_data, model_name=model_name, gen_prompt=gen_prompt, w_file=MGSM_gen_file, max_new_tokens=max_new_tokens)
        elif model_name == 'RedPajama-INCITE-Base-3B-v1':
            redpajama(data=mgsm_data, model_name=model_name, gen_prompt=gen_prompt, w_file=MGSM_gen_file, max_new_tokens=max_new_tokens)
    else:
        # evaluate
        from evaluate_05 import eval_MGSM
        new_data = load_MGSM(MGSM_gen_file, samples_num=100)
        eval_MGSM(data=new_data, model=model_name, w_file=MGSM_gen_file, metrics='maj@k')


def eval():
    # # MGSM
    # altering = ['', 'ex_adjacent', 'ex_random_two']
    run_MGSM(lan='fr', model_name='llama-7b', altering='', max_new_tokens=-1, evaluate=True)
    run_MGSM(lan='fr', model_name='llama-7b', altering='_ex_random_two', max_new_tokens=-1, evaluate=True)
    run_MGSM(lan='fr', model_name='llama-7b', altering='_ex_adjacent', max_new_tokens=-1, evaluate=True)
    run_MGSM(lan='fr', model_name='llama-7b', altering='_rotate_two_part', max_new_tokens=-1, evaluate=True)

    run_MGSM(lan='fr', model_name='llama-13b', altering='', max_new_tokens=-1, evaluate=True)
    run_MGSM(lan='fr', model_name='llama-13b', altering='_ex_random_two', max_new_tokens=-1, evaluate=True)
    run_MGSM(lan='fr', model_name='llama-13b', altering='_ex_adjacent', max_new_tokens=-1, evaluate=True)
    run_MGSM(lan='fr', model_name='llama-13b', altering='_rotate_two_part', max_new_tokens=-1, evaluate=True)

    run_MGSM(lan='fr', model_name='llama-30b', altering='', max_new_tokens=-1, evaluate=True)
    run_MGSM(lan='fr', model_name='llama-30b', altering='_ex_random_two', max_new_tokens=-1, evaluate=True)
    run_MGSM(lan='fr', model_name='llama-30b', altering='_ex_adjacent', max_new_tokens=-1, evaluate=True)
    run_MGSM(lan='fr', model_name='llama-30b', altering='_rotate_two_part', max_new_tokens=-1, evaluate=True)



if __name__ == "__main__":
    # eval()
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default='llama-7b')
    args.add_argument("--altering", type=str, default='')
    args.add_argument("--lan", type=str, default='fr')
    args.add_argument('--task', type=str, default='TruthfulQA')
    args = args.parse_args()

    run_MGSM(lan=args.lan, model_name=args.model_name, altering=args.altering, max_new_tokens=-1, evaluate=False)

