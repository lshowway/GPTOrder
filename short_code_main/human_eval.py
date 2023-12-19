import json
from random import shuffle

fw_a = open('D:/phd5/KR/natural_questions_human_a.txt', 'w', encoding='utf-8')
fw_b = open('D:/phd5/KR/natural_questions_human_b.txt', 'w', encoding='utf-8')
fw_c = open('D:/phd5/KR/natural_questions_human_c.txt', 'w', encoding='utf-8')
fw_I = open('D:/phd5/KR/natural_questions_human_I.txt', 'w', encoding='utf-8')
fw_II = open('D:/phd5/KR/natural_questions_human_II.txt', 'w', encoding='utf-8')

with open('D:/phd5/KR/natural_questions.jsonl', 'r', encoding='utf-8') as f:
    all_questions, selected_questions = [], []
    for line in f:
        json_obj = json.loads(line)
        all_questions.append(json_obj)
    selected_questions = all_questions[:10] + all_questions[-10:]

for x in selected_questions:
    id = x['id']
    length = x['length']
    question = x['question']
    q_list = question.split()

    # a. exhange the first and last word
    a_question = q_list[-1] + ' ' + ' '.join(q_list[1:-1]) + ' ' + q_list[0]
    json.dump({"id": id, "length": length, "prompt": a_question}, fw_a)
    fw_a.write("\nPlease reorder the sentence into normal:" + '\n')

    # b. exchange adjacent words
    new_i = []
    for i in range(len(q_list) // 2):
        t = [2*i+1, 2*i]
        new_i.extend(t)
    if len(q_list) % 2 == 1:
        new_i.append(len(q_list) - 1)
    b_question = ' '.join([q_list[i] for i in new_i])
    json.dump({"id": id, "length": length, "prompt": b_question}, fw_b)
    fw_b.write("\nPlease reorder the sentence into normal:" + '\n')

    # c. fix the first and last word, shuffle others
    t2 = q_list[1:-1]
    shuffle(t2)
    c_question = q_list[0] + ' ' + ' '.join(t2) + ' ' + q_list[-1]
    json.dump({"id": id, "length": length, "prompt": c_question}, fw_c)
    fw_c.write("\nPlease reorder the sentence into normal:" + '\n')

    # I. shuffle in trigram
    new_j = []
    for i in range(len(q_list) // 3):
        j = [3*i, 3*i+1, 3*i+2]
        shuffle(j)
        new_j.extend(j)
    if len(q_list) % 3 == 1:
        new_j.append(len(q_list) - 1)
    elif len(q_list) % 3 == 2:
        new_j.extend([len(q_list) - 2, len(q_list) - 1])
    I_question = ' '.join([q_list[i] for i in new_j])
    json.dump({"id": id, "length": length, "prompt": I_question}, fw_I)
    fw_I.write("\nPlease reorder the sentence into normal:" + '\n')

    # II. reverse the order of words
    II_question = ' '.join(q_list[::-1])
    json.dump({"id": id, "length": length, "prompt": II_question}, fw_II)
    fw_II.write("\nPlease reorder the sentence into normal:" + '\n')


fw_a.close()
fw_b.close()
fw_c.close()
fw_I.close()
fw_II.close()