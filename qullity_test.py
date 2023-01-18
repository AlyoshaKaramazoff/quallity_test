import os
import json
import math
import numpy as np
import pandas as pd
import csv
from collections import Counter, OrderedDict
from utils import *
import textdistance as textdistance
from transliterate import translit
from transliterate.base import TranslitLanguagePack, registry


class ExampleLanguagePack(TranslitLanguagePack):
    language_code = "example"
    language_name = "Example"
    mapping = (
    u"abvgdezijklmnoprstufhcC'y'ABVGDEZIJKLMNOPRSTUFH'Y'",
    u"абвгдезийклмнопрстуфхцЦъыьАБВГДЕЗИЙКЛМНОПРСТУФХЪЫЬ",
    )

    reversed_specific_mapping = (
        u"ёэЁЭъьЪЬ",
        u"eeEE''''"
    )

    pre_processor_mapping = {
        u"zh": u"ж",
        u"ts": u"ц",
        u"ch": u"ч",
        u"sh": u"ш",
        u"shh": u"щ",
        u"ju": u"ю",
        u"ja": u"я",
        u"Zh": u"Ж",
        u"Ts": u"Ц",
        u"Ch": u"Ч",
        u"Sh": u"Ш",
        u"Sch": u"Щ",
        u"Ju": u"Ю",
        u"Ja": u"Я",
        u"_00_": u"ѱ",
        u"_01_": u"ї",
        u"_02_": u"ѵ",
        u"_03_": u"æ",
        u"_04_": u"Æ",
        u"*": u"ѣ",
        u"_10_": u"ꙓ",
        u"_05_": u"Ⱚ",
        u"_06_": u"Ѳ",
        u"_07_": u"Ⱇ",
        u"_08_": u"љ",
        u"_09_": u"ѩ",
        u"_11_": u"ѷ",
        u"___": u"-",
         u"_i": u"i",
         u"ya":u"ья",
         u"yu":u"ью",
         u"chya":u"чья",
         u"chk":u"чьк",
         u"lshh":u"льщ",
         u"lsh":u"льш",
         u"yach":u"ячь",
         u"zhya":u"жья"
    }
registry.register(ExampleLanguagePack)

# file with segments
with open(input_seg_file, 'r', encoding='utf-8') as file:
    dataSeg = json.load(file)

# file with result
with open(input_result_file, 'r', encoding='utf-8') as file:
    dataRes = json.load(file)

body_words = dataRes['body_words']

PositionAll = {}
for item in body_words:
     PositionAll.setdefault(item['predicted_text'], [])
     PositionAll[item['predicted_text']].append(item['coords'])
PositionAllCopy = PositionAll

segPositionAll = {}
for item in dataSeg['selectedList']:
    key = translit(item['name'], "example").lower()
    segPositionAll.setdefault(key, [])
    segPositionAll[key].append(item['rect'])

counterAll = Counter()
counterAllTP = Counter()
counterAllFP = Counter()
counterAllFN = Counter()
counterAllTC = Counter()
counterAllFC = Counter()
counterAllFC_hamDist = Counter()

m_index = []
i = 0
for key in segPositionAll.keys():

    indexS = i
    segWord = key
    listValuesS = segPositionAll[key]

    for k in listValuesS:
        k_centr = ((k['x'] + k['width']) / 2, (k['y'] + k['height']) / 2)
        min_dist = 1000000

        j = 0
        for key in PositionAll.keys():
            indexP = j
            listValuesP = PositionAll[key][0]
            predWord = key


            # координаты определённого сетью слова
            m_x = listValuesP['x']
            m_y = listValuesP['y']
            m_width = listValuesP['width']
            m_height = listValuesP['height']

            m_centr = ((m_x + m_width) / 2, (m_y + m_height) / 2)
            distance = math.sqrt(np.dot(k_centr[0] - m_centr[0], k_centr[0] - m_centr[0]) + np.dot(k_centr[1] - m_centr[1], k_centr[1] - m_centr[1]))

            # находим самое близкое по расстоянию слово среди определённых сетью к сегменту
            if min_dist > distance:
                min_dist = distance
                m_detect = listValuesP
                m_idx = indexP
                m_word = predWord

            # переход к след. элементу PositionAll
            j += 1

        m_index.append(m_idx)

        rectSeg = [k['x'], k['y'], k['x'] + k['width'], k['y'] + k['height']]
        rectDetect = [m_detect['x'], m_detect['y'], m_detect['x'] + m_detect['width'], m_detect['y'] + m_detect['height']]
        iou = bb_intersection_over_union(rectSeg, rectDetect)

        if iou >= 0.5:
            counterAllTP[list(segPositionAll.keys())[indexS]] += 1  # collections objects for True Positive
            hamDistance = textdistance.hamming(segWord, m_word)
            if hamDistance == 0:
                counterAllTC[list(segPositionAll.keys())[indexS]] += 1   # collections objects for True Classified
            else:
                counterAllFC[list(segPositionAll.keys())[indexS]] += 1   # collections objects for False Classified
                counterAllFC_hamDist[list(segPositionAll.keys())[indexS]] += hamDistance
                # print(f"FC - {list(segPositionAll.keys())[indexS]} - hamDist: {hamDistance}")
        else:
            counterAllFN[list(segPositionAll.keys())[indexS]] += 1     # collections objects for False Negative

    # переход к след. элементу segPositionAll
    i += 1


# collections objects for False Positive
positionAll = [x for ind, x in enumerate(PositionAllCopy) if ind not in m_index]  # здесь лежат те прямоугольники, которые не попали в сравнение
for item in positionAll:
    counterAllFP[item] += 1

# создаем временный словарь в котором будут лежать все уникальные ключи
tempDict = {}
tempDict.update(counterAllTP)
tempDict.update(counterAllFN)
tempDict.update(counterAllFP)
for label in tempDict.keys():
    TP = counterAllTP.get(label, 0)
    FN = counterAllFN.get(label, 0)
    FP = counterAllFP.get(label, 0)
    TC = counterAllTC.get(label, 0)
    FC = counterAllFC.get(label, 0)
    ham_dist = counterAllFC_hamDist.get(label, 0)
    tempMetrics = Counter({"TP": TP, "FN": FN, "FP": FP, "TC": TC, "FC": FC, "HamDist": ham_dist})
    if counterAll.get(label) == None:
        counterAll.setdefault(label, {})
    counterAll[label] = Counter(counterAll[label]) + tempMetrics


TP = 0
FP = 0
FN = 0
TC = 0
FC = 0

os.chdir(output_folder)
file_out = open("temp.csv", 'w', newline='')
writer = csv.writer(file_out, delimiter=',')
writer.writerow(("Product Name", "Sample", "GT", "TP", "FN", "FP", "TC", "FC", "Miss", "HamDist"))


for word, metrics in counterAll.items():
    evaluation = {}
    for key, value in dict(metrics).items():
        evaluation.update({key: value})  # объединение словарей с оценками в один словарь
    tp = evaluation.get("TP", 0)
    fn = evaluation.get("FN", 0)
    fp = evaluation.get("FP", 0)
    tc = evaluation.get("TC", 0)
    fc = evaluation.get("FC", 0)
    hamDist = evaluation.get("HamDist", 0)
    TP += tp
    FN += fn
    FP += fp
    TC += tc
    FC += fc
    writer.writerow([word, "Summary", tp+fn, tp, fn, fp, tc, fc, hamDist])
    writer.writerow([word, "Summary, %", '100', tp/(tp+fn)*100 if tp+fn > 0 else 0, fn/(tp+fn)*100 if tp+fn > 0 else 0, '', tc/(tc+fc)*100 if tc+fc > 0 else 0, fc/(tc+fc)*100 if tc+fc > 0 else 0, ''])

precision = TP / (TP + FP)
recall = TP / (TP + FN)

file_out.close()

df = pd.read_csv("temp.csv")
df = df.fillna(value='')
df = df.sort_values(by=['Product Name', 'Sample'])

d = [['', '', '', '', '', '', '', '', '', ''],
    ["Precision", precision, '', '', '', '', '', '', '', ''],
    ["Recall", recall, '', '', '', '', '', '', '', ''],
    ["Overall FP", FP, '', '', '', '', '', '', '', ''],
    ["True Classified", TC, '', '', '', '', '', '', '', ''],
    ["False Classified", FC, '', '', '', '', '', '', '', ''],
    ["True Classified %", TC/(TC+FC) * 100, '', '', '', '', '', '', '', ''],
    ["False Classified %", FC/(TC+FC) * 100, '', '', '', '', '', '', '', '']]

new = pd.DataFrame(d, columns=df.columns)
df = df.append(new)
df.to_csv('result.csv', index=False)


# print(counterAllTP)
# print(counterAllTC)
# print(counterAllFC)
# print(counterAllFN)
# print(counterAllFP)

