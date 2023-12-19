import json

q_file = "D:\phd5\KR/natural_questions.jsonl"
# chat_reorder_file = "D:\phd5\KR/chat_reorder/natural_questions_human_II.txt"
chat_reorder_file = "D:\phd5\KR/human_reorder/natural_questions_human_I.txt"

# read chat answers file
chat_reorders = []
with open(chat_reorder_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith("Please reorder the sentence into normal:"):
            t = line.split(":")[1].strip().lower()
            t = t.replace('.', '').replace('?', '')
            chat_reorders.append(t.split())

# read q file
questions = []
with open(q_file, 'r', encoding='utf-8') as f:
    for line in f:
        json_obj = json.loads(line)
        q = json_obj['question'].lower()
        questions.append(q.split())

# 1. compute BLUE score
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
blue_list_1, blue_list_2, blue_list_3, blue_list_4 = [], [], [], []
for q, re in zip(questions, chat_reorders):
    blue_value_1 = sentence_bleu([q], re, weights=(1, 0, 0, 0))
    blue_value_2 = sentence_bleu([q], re, weights=(0, 1, 0, 0))
    blue_value_3 = sentence_bleu([q], re, weights=(0, 0, 1, 0))
    blue_value_4 = sentence_bleu([q], re, weights=(0, 0, 0, 1))

    blue_list_1.append(blue_value_1)
    blue_list_2.append(blue_value_2)
    blue_list_3.append(blue_value_3)
    blue_list_4.append(blue_value_4)

blue_list_1 = np.array(blue_list_1)
blue_list_2 = np.array(blue_list_2)
blue_list_3 = np.array(blue_list_3)
blue_list_4 = np.array(blue_list_4)

# print("blue-1 score: ", blue_list_1)
print("mean blue-1 score: ", round(np.mean(blue_list_1), 4))

# print("blue-2 score: ", blue_list_2)
print("mean blue-2 score: ", round(np.mean(blue_list_2), 4))

# print("blue-3 score: ", blue_list_3)
print("mean blue-3 score: ", round(np.mean(blue_list_3), 4))

# print("blue-4 score: ", blue_list_4)
print("mean blue-4 score: ", round(np.mean(blue_list_4), 4))


