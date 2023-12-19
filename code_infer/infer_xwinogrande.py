import time

import torch


from load_data_02 import load_XWinoGrade
from prompt import winogrande_generation_prompt
from evaluate_05 import eval_XWinoGrade

global_devices = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]
max_memory = {k: '40GB' for k in global_devices}


def llama_hugging(lan, data, model_name='llama-7b', winogrande_generation_prompt=None, w_file=None,
                  temperature=None, max_new_tokens=None, k=1):
    from transformers.models.llama import LlamaTokenizer
    from transformers import AutoModelForCausalLM
    llama_model = AutoModelForCausalLM.from_pretrained('./foundation_models/{}'.format(model_name),
                                                   low_cpu_mem_usage=True, device_map='balanced',
                                                   torch_dtype=torch.float32, max_memory=max_memory)
    tokenizer = LlamaTokenizer.from_pretrained('./foundation_models/{}'.format(model_name))

    if model_name not in data.columns:
        data[model_name] = ''
        data[model_name].fillna('', inplace=True)
        data[model_name] = data[model_name].astype(str)

    for index, row in data.iterrows():
        label = row['answer']
        question = row['Question']
        option1 = row['option1']
        option2 = row['option2']

        if lan == 'zh':
            Q = "{}选项1: {}, 选项2: {}".format(question, option1, option2)
        else:
            Q = "{} option1: {}, option2: {}".format(question, option1, option2)
        # else:
        #     Q = None
        #     print('input language')
        prompt = winogrande_generation_prompt[lan].format(Q)
        k_answers = []
        start = time.time()
        count = 0
        while count < k:
            inputs = tokenizer(prompt, return_tensors="pt")
            generate_ids = llama_model.generate(inputs.input_ids.to(llama_model.device),
                                                max_new_tokens=max_new_tokens, top_k=10, top_p=0.9, do_sample=True,
                                                temperature=temperature, early_stopping=True,
                                                use_cache=True
                                                )
            answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            if lan == 'zh':
                answer = answer.split('选项2: 李梅亭')[-1].split('\n')
                for a in answer:
                    a = a.strip()
                    if a.startswith('A:'):
                        k_answers.append(a)
                        print('>>', a)
                        count += 1
                        break
            elif lan == 'en':
                answer = answer.split('A: option2: The criminal')[-1].split('\n')
                print(answer)

                for a in answer:
                    a = a.strip()
                    if a.startswith('A:'):
                        k_answers.append(a)
                        print('>>', a)
                        count += 1
                        break
            elif lan == 'fr':
                answer = answer.split('A: option2: gilet')[-1].split('\n')
                print(answer)
                for a in answer:
                    if a.startswith('A'):
                        k_answers.append(a)
                        print('>>', a)
                        count += 1
                        break
        data.loc[index, model_name] = str(k_answers)
        print('=================', time.time() - start, w_file.split('/')[-1], model_name, Q, label)
        if (index + 1) % 5 == 0:
            data.to_csv(w_file, index=False, encoding='utf-8')
    data.to_csv(w_file, index=False, encoding='utf-8')


def run_XWinogrande(lan='en', model_name='llama-7b', altering='_ex_random_two',
                    temperature=0.5, max_new_tokens=None, k=1, evaluate=True):
    xwinogrande_file = "../data/XWinoGrande/xwinogrande_{}{}.csv".format(lan, altering)
    xwinogrande_gen_file = "../data/XWinoGrande/{}_xwinogrande_{}{}.csv".format(model_name, lan, altering)

    xwinogrande_data = load_XWinoGrade(xwinogrande_file)[:100]
    # gen_prompt = winogrande_generation_prompt

    if not evaluate:
        if model_name in ['llama-7b', 'llama-13b', 'llama-30b', 'llama-65b']:
            llama_hugging(lan=lan, data=xwinogrande_data, model_name=model_name,
                          winogrande_generation_prompt=winogrande_generation_prompt,
                          w_file=xwinogrande_gen_file, temperature=temperature, max_new_tokens=max_new_tokens, k=k)
    else:
        # evaluate
        new_data = load_XWinoGrade(xwinogrande_gen_file, samples_num=100)
        eval_XWinoGrade(data=new_data, model=model_name, w_file=xwinogrande_gen_file, metrics='acc')


def eval():
    run_XWinogrande(lan='zh', model_name='llama-7b', altering='', max_new_tokens=-1, evaluate=True)
    run_XWinogrande(lan='zh', model_name='llama-7b', altering='_ex_random_two', max_new_tokens=-1, evaluate=True)
    run_XWinogrande(lan='zh', model_name='llama-7b', altering='_rotate_two_part', max_new_tokens=-1, evaluate=True)
    run_XWinogrande(lan='zh', model_name='llama-7b', altering='_ex_adjacent', max_new_tokens=-1, evaluate=True)

    # run_XWinogrande(lan='en', model_name='llama-30b', altering='', max_new_tokens=10, evaluate=True)
    # run_XWinogrande(lan='en', model_name='llama-30b', altering='_ex_random_two', max_new_tokens=10, evaluate=True)
    # run_XWinogrande(lan='en', model_name='llama-30b', altering='_rotate_two_part', max_new_tokens=10, evaluate=True)
    # run_XWinogrande(lan='en', model_name='llama-30b', altering='_ex_adjacent', max_new_tokens=10, evaluate=True)



if __name__ == "__main__":
    eval()

    # import argparse
    #
    # args = argparse.ArgumentParser()
    # args.add_argument("--model_name", type=str, default='llama-7b')
    # args.add_argument("--altering", type=str, default='')
    # args.add_argument("--lan", type=str, default='en')
    # args.add_argument('--task', type=str, default='TruthfulQA')
    #
    # args = args.parse_args()
    # run_XWinogrande(lan=args.lan, model_name=args.model_name, altering=args.altering,
    #                 max_new_tokens=50, k=1, evaluate=False)