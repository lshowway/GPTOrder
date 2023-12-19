import os.path


from load_data_02 import load_TruthfulQA
from generation_04 import generate_TruthfulQA
from evaluate_05 import eval_TruthfulQA

from load_data_02 import load_MGSM
from generation_04 import generate_MGSM
from evaluate_05 import eval_MGSM


def run_ChatGPT_TruthfulQA(model='text-davinci-003', altering=''):
    raw_file = "../data/TruthfulQA/TruthfulQA{}.csv".format(altering)
    generated_file = "../data/TruthfulQA/ChatGPT_TruthfulQA{}.csv".format(altering)

    data = load_TruthfulQA(raw_file, samples_num=100)
    if not os.path.exists(generated_file):
        data.to_csv(generated_file, index=False)
    data = load_TruthfulQA(generated_file, samples_num=100)
    # generate_TruthfulQA(data=data, model=model, w_file=generated_file, temperature=0.0, max_tokens=50, top_k=1)
    new_data = load_TruthfulQA(generated_file)
    # fine-tuned by 1000 samples
    eval_TruthfulQA(data=new_data, g_model=model, ft_model='curie:ft-personal-2023-05-08-10-18-37', w_file=generated_file, metrics='GPT-judge')
    eval_TruthfulQA(data=new_data, g_model=model, ft_model='curie:ft-personal-2023-05-08-10-20-38',  w_file=generated_file, metrics='GPT-info')
    # eval_TruthfulQA(data=new_data, g_model=model, ft_model='', w_file=generated_file, metrics='bleurt')


def run_ChatGPT_MGSM(model='gpt-3.5-turbo', lan='en', altering=''):
    raw_file = "../data/mgsm/mgsm_{}{}.csv".format(lan, altering)
    generated_file = "../data/mgsm/ChatGPT_GSM_{}{}.csv".format(lan, altering)
    if os.path.exists(generated_file):
        data = load_MGSM(generated_file, samples_num=100)
    else:
        data = load_MGSM(raw_file, samples_num=100)
    # generate_MGSM(data=data, lan=lan, model=model, w_file=generated_file, temperature=0.0, max_tokens=128, top_k=1, k=1)
    new_data = load_MGSM(generated_file, samples_num=100)
    eval_MGSM(data=new_data, model=model, w_file=generated_file, metrics='maj@k')


def run_chatGPT_WinoGrande(model='gpt-3.5-turbo', lan='zh', altering=''):
    from load_data_02 import load_XWinoGrade
    from generation_04 import generate_XWinoGrade
    from evaluate_05 import eval_XWinoGrade

    raw_file = "../data/XWinoGrande/xwinogrande_{}{}.csv".format(lan, altering)
    generated_file = "../data/XWinoGrande/ChatGPT_xwinograde_{}{}.csv".format(lan, altering)

    data = load_XWinoGrade(raw_file, samples_num=100)
    if not os.path.exists(generated_file):
        data.to_csv(generated_file, index=False)
    data = load_XWinoGrade(generated_file, samples_num=100)
    # generate_XWinoGrade(data=data, lan=lan, model=model, w_file=generated_file, temperature=0.0, max_tokens=20, top_k=1, k=1)
    new_data = load_XWinoGrade(generated_file, samples_num=100)
    eval_XWinoGrade(data=new_data, model=model, w_file=generated_file, metrics='acc')


def run_Chat_WiQueen(model='text-davinci-003', lan='zh', altering=''):
    from load_data_02 import load_WiQueen
    from generation_04 import generate_WiQueen
    from evaluate_05 import eval_WiQueen

    raw_file = "../data/WiQueen/analogy_unique_{}{}.csv".format(lan, altering)
    generated_file = "../data/WiQueen/Chat_analogy_unique_{}{}.csv".format(lan, altering)

    data = load_WiQueen(raw_file, samples_num=100)
    # generate_WiQueen(data=data, lan=lan, model=model, w_file=generated_file, temperature=0.0, max_tokens=10, top_k=1, k=1)
    new_data = load_WiQueen(generated_file, samples_num=100)
    eval_WiQueen(data=new_data, model=model, w_file=generated_file, metrics='P@1')


def run_Claude_TruthfulqQA(altering=''):
    raw_file = "../data/TruthfulQA/TruthfulQA{}.csv".format(altering)
    generated_file = "../data/TruthfulQA/Claude_TruthfulQA{}.csv".format(altering)

    data = load_TruthfulQA(raw_file, samples_num=100)
    if not os.path.exists(generated_file):
        data.to_csv(generated_file, index=False)
    data = load_TruthfulQA(generated_file, samples_num=100)
    # generate_TruthfulQA(data=data, model='claude-v1.3', w_file=generated_file,
    #                     temperature=0.0, max_tokens=50, top_k=1)
    new_data = load_TruthfulQA(generated_file)
    eval_TruthfulQA(data=new_data, g_model='claude-v1.3', ft_model='curie:ft-personal-2023-05-08-10-18-37', w_file=generated_file, metrics='GPT-judge')
    eval_TruthfulQA(data=new_data, g_model='claude-v1.3', ft_model='curie:ft-personal-2023-05-08-10-20-38', w_file=generated_file, metrics='GPT-info')
    # eval_TruthfulQA(data=new_data, g_model='claude-v1.3', ft_model='', w_file=generated_file, metrics='bleurt')


def run_Claude_MGSM(model='claude-v1.3', lan='en', altering=''):
    raw_file = "../data/mgsm/mgsm_{}{}.csv".format(lan, altering)
    generated_file = "../data/mgsm/Claude_MGSM_{}{}.csv".format(lan, altering)

    data = load_MGSM(raw_file, samples_num=100)
    # generate_MGSM(data=data, lan=lan, model=model, w_file=generated_file, temperature=0.0, max_tokens=128, top_k=1, k=1)
    new_data = load_MGSM(generated_file, samples_num=100)
    eval_MGSM(data=new_data, model='claude-v1.3', w_file=generated_file, metrics='acc')


def run_Claude_XWinoGrade(model='claude-v1.3', lan='zh', altering=''):
    from load_data_02 import load_XWinoGrade
    from generation_04 import generate_XWinoGrade
    from evaluate_05 import eval_XWinoGrade

    raw_file = "../data/XWinoGrande/xwinogrande_{}{}.csv".format(lan, altering)
    generated_file = "../data/XWinoGrande/Claude_xwinograde_{}{}.csv".format(lan, altering)

    data = load_XWinoGrade(raw_file, samples_num=100)
    # generate_XWinoGrade(data=data, lan=lan, model=model, w_file=generated_file, temperature=0.0, max_tokens=20, top_k=1, k=1)
    new_data = load_XWinoGrade(generated_file, samples_num=100)
    eval_XWinoGrade(data=new_data, model='claude-v1.3', w_file=generated_file, metrics='acc')


def run_Claude_WiQueen(model='claude-v1.3', lan='zh', altering=''):
    from load_data_02 import load_WiQueen
    from generation_04 import generate_WiQueen
    from evaluate_05 import eval_WiQueen

    raw_file = "../data/WiQueen/analogy_unique_{}{}.csv".format(lan, altering)
    generated_file = "../data/WiQueen/Claude_analogy_unique_{}{}.csv".format(lan, altering)

    data = load_WiQueen(raw_file, samples_num=100)
    generate_WiQueen(data=data, lan=lan, model=model, w_file=generated_file, temperature=0.0, max_tokens=10, top_k=1, k=1)
    new_data = load_WiQueen(generated_file, samples_num=100)
    eval_WiQueen(data=new_data, model='claude-v1.3', w_file=generated_file, metrics='P@1')



if __name__ == "__main__":
    from transformers.models.bert.modeling_bert import BertForMaskedLM
    # TruthfulQA
    # run_ChatGPT_TruthfulQA(altering='')
    # run_ChatGPT_TruthfulQA(altering='_ex_random_two')
    # run_ChatGPT_TruthfulQA(altering='_rotate_two_part')
    # run_ChatGPT_TruthfulQA(altering='_ex_adjacent')

    # run_Claude_TruthfulqQA()
    # run_Claude_TruthfulqQA(altering='_ex_random_two')
    # run_Claude_TruthfulqQA(altering='_rotate_two_part')
    # run_Claude_TruthfulqQA(altering='_ex_adjacent')

    # MGSM     # altering = ['', '_ex_adjacent', '_ex_random_two']
    run_ChatGPT_MGSM(lan='en')
    run_ChatGPT_MGSM(lan='en', altering='_ex_random_two')
    run_ChatGPT_MGSM(lan='en', altering='_rotate_two_part')
    run_ChatGPT_MGSM(lan='en', altering='_ex_adjacent')

    run_Claude_MGSM(lan='en')
    run_Claude_MGSM(lan='en', altering='_ex_random_two')
    run_Claude_MGSM(lan='en', altering='_rotate_two_part')
    run_Claude_MGSM(lan='en', altering='_ex_adjacent')

    # run_Claude_XWinoGrade(lan='fr', altering='')
    # run_Claude_XWinoGrade(lan='fr', altering='_ex_random_two')
    # run_Claude_XWinoGrade(lan='fr', altering='_rotate_two_part')
    # run_Claude_XWinoGrade(lan='fr', altering='_ex_adjacent')

    # run_chatGPT_WinoGrande(model='text-davinci-003', lan='fr', altering='')
    # run_chatGPT_WinoGrande(model='text-davinci-003', lan='fr', altering='_ex_random_two')
    # run_chatGPT_WinoGrande(model='text-davinci-003', lan='fr', altering='_rotate_two_part')
    # run_chatGPT_WinoGrande(model='text-davinci-003', lan='fr', altering='_ex_adjacent')

    # 6.1
    # run_Claude_WiQueen(model='claude-v1.3', lan='fr', altering='')
    # run_Claude_WiQueen(model='claude-v1.3', lan='fr', altering='_ex_random_two')
    # run_Claude_WiQueen(model='claude-v1.3', lan='fr', altering='_rotate_two_part')
    # run_Claude_WiQueen(model='claude-v1.3', lan='fr', altering='_ex_adjacent')

    # run_Chat_WiQueen(model='text-davinci-003', lan='fr', altering='')
    # run_Chat_WiQueen(model='text-davinci-003', lan='fr', altering='_ex_random_two')
    # run_Chat_WiQueen(model='text-davinci-003', lan='fr', altering='_rotate_two_part')
    # run_Chat_WiQueen(model='text-davinci-003', lan='fr', altering='_ex_adjacent')



