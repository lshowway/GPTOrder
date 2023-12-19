"""
preprocess the data we will use, convert all of them into csv format

"""
import pandas as pd
import json
import pandas as pd
import re


def convert_sst():
    # it is implemented directly by excel
    pass


def convert_wino():
    # convert jsonl to csv

    file_path = "../data/XWinoGrande/keep/fr.jsonl"
    data = []

    with open(file_path, "r", encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame as a CSV file
    csv_file_path = "../data/XWinoGrande/xwinogrande_fr.csv"
    df.to_csv(csv_file_path, index=False, columns=["sentence", "option1", "option2", 'answer', 'lang'], encoding='utf-8')


def convert_gsm8k():
    # it is implemented directly by excel
    pass


def convert_truthfulqa():
    # No need to convert, it is in csv file
    pass

def convert_wiqueen():
    file = "../data/WiQueen/keep/analogy_unique_fr_contexts.csv.test"
    w_file = "../data/WiQueen/analogy_unique_fr.csv"

    # e1, e2, e3, e4_candidates
    all_lines = []
    low_list = []
    with open(file, encoding='utf-8') as fr:
        for line in fr:
            if line.startswith('#'):
                pattern = r"\((P\d+)\)"
                match1 = re.search(pattern, line)
                if match1:
                    relation = match1.group(1)
                    print(relation)
            else:
                line = line.strip()
                line = line.split(';')
                e1, q1, info1, e2, q2, info2, e3, q3, info3, e4, q4, info4, _, _ = line
                print(info4)
                try:
                    info4 = eval(info4)
                    if info4['aliases'] is not None:
                        print(info4)
                        e4_candidates = [e4]  + info4['aliases']
                        if len(e4_candidates) < 4:
                            low_list.append([relation, e1, e2, e3, str(e4_candidates)])
                        else:
                            all_lines.append([relation, e1, e2, e3, str(e4_candidates)])
                except:
                    pass
    all_lines.extend(low_list)
    print(all_lines)
    # Create a DataFrame from the list
    df = pd.DataFrame(all_lines, columns=['relation', 'e1', 'e2', 'e3', 'e4_candidates'])

    # Save DataFrame to a CSV file
    df.to_csv(w_file, index=False, encoding='utf-8')


if __name__ == "__main__":
    # convert_wino()
    # from may_3_load_data import load_TruthfulQA
    # data = load_TruthfulQA('../data/TruthfulQA/TruthfulQA.csv')
    # t = data['Question'].tolist()
    #
    # word_number = []
    # for x in t:
    #     word_number.append(len(x.split()))
    # print(sum(word_number) / len(word_number))

    from load_data_02 import load_WiQueen
    data = load_WiQueen('../data/WiQueen/analogy_unique_fr.csv')
    t1 = data['e1'].tolist()
    t2 = data['e2'].tolist()
    t3 = data['e3'].tolist()
    word_number = []

    for e1, e2, e3 in zip(t1, t2, t3):
        x = e1.split() + e2.split() + e3.split()
        word_number.append(len(x))

    print(sum(word_number) / len(word_number) / 3)