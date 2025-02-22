import os
import pickle as pkl
import numpy as np
import json
import time

import pandas as pd
import nltk
from konlpy.tag import Okt

import torch
from keras.preprocessing.sequence import pad_sequences

okt = Okt()


def tokenize(doc):
    """
    input:
        doc: dtype=string, example) '아 더빙.. 진짜 짜증나네요 목소리'
    return:
        list: dtype=string, elements example) ['아/Exclamation', '더빙/Noun', '../Punctuation', '진짜/Noun', '짜증나다/Adjective', '목소리/Noun']
    """
    return ["/".join(token) for token in okt.pos(doc, norm=True, stem=True)]


def saveDocs(docs, filePath):
    # remove duplicate samples.
    docs.drop_duplicates(subset=["document"], inplace=True)

    # remove all characters except korean and spaces.
    docs["document"] = docs["document"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")

    # exchange spaces value to null value.
    docs["document"].replace("", np.nan, inplace=True)

    # remove null samples.
    docs = docs.dropna(how="any")

    # reviewDocs has [("morpheme1/tag1", "morpheme2/tag2", ... ,), label(0 or 1)]
    reviewDocs = [(tokenize(row[1]), row[2]) for row in docs.values]

    with open(filePath, "wb") as f:
        pkl.dump(reviewDocs, f)


def loadDocs(filePath):
    with open(filePath, "rb") as f:
        reviewDocs = pkl.load(f)
        return reviewDocs


def frequency2index(jsonDict, topN):
    lookupTable = dict()
    lookupTable["<pad>"] = 0  # pad value to match the maximum length.
    lookupTable["<oov>"] = 1  # out of value (if a word not in lookup table's keys.)
    i = 2
    for key in jsonDict.keys():
        lookupTable[key] = i
        i += 1

    return lookupTable


def saveTopNwords(dataTokens, topN, filePath):
    text = nltk.Text(dataTokens, name="NMSC")

    # get TOP N tokens with the highest frequency of output.
    topN_words = {token[0]: token[1] for token in text.vocab().most_common(topN)}

    with open(filePath, "w", encoding="utf-8") as jsonF:
        json.dump(topN_words, jsonF, ensure_ascii=False)


def loadTopNwords(filePath):
    with open(filePath, "r", encoding="utf-8") as jsonF:
        topN_words = json.load(jsonF)

    return topN_words


def saveTopNindex(topN_words, topN, filePath):
    topN_index = frequency2index(topN_words, topN=topN)
    with open(filePath, "w", encoding="utf-8") as jsonF:
        json.dump(topN_index, jsonF, ensure_ascii=False)


def loadTopNindex(filePath):
    with open(filePath, "r", encoding="utf-8") as jsonF:
        topN_index = json.load(jsonF)
        return topN_index


def encodeToInt(train_x, lookupTable, maxLen):
    encodeIntList = []
    keySet = lookupTable.keys()
    for x in train_x:
        encodeIntLine = []
        for token in x:
            if token in keySet:
                encodeIntLine.append(lookupTable[token])
            else:
                encodeIntLine.append(lookupTable["<oov>"])
        encodeIntList.append(encodeIntLine)

    encodeIntList = np.array(encodeIntList, dtype="object")
    padEncodeIntList = pad_sequences(encodeIntList, maxlen=maxLen, padding="pre", value=lookupTable["<pad>"])
    return padEncodeIntList

    import torch


if __name__ == "__main__":
    if not os.path.isfile("./dataSet/trainDocs.pkl"):
        train_df = pd.read_csv("./nsmc-master/ratings_train.txt", "\t")
        saveDocs(train_df, "./dataSet/trainDocs.pkl")
    if not os.path.isfile("./dataSet/testDocs.pkl"):
        test_df = pd.read_csv("./nsmc-master/ratings_test.txt", "\t")
        saveDocs(test_df, "./dataSet/testDocs.pkl")
    if not os.path.isfile("./dataSet/dataDocs.pkl"):
        data_df = pd.read_table("./nsmc-master/ratings.txt")
        saveDocs(data_df, "./dataSet/dataDocs.pkl")

    # ==================================================================================================
    # ========= process that get a Top N lookup table from comprehensive data set[ratings.txt] =========
    # ==================================================================================================

    """
    topN 과 maxLen을 변경하여 데이터 셋의 전처리를 변경할 수 있습니다.
    """
    topN = 10000
    maxLen = 80

    if not os.path.isfile("./dataSet/top_%d_maxLen_%d_words.json" % (topN, maxLen)) and not os.path.isfile("./dataSet/top_%d_maxLen_%d_index.json" % (topN, maxLen)):
        dataDocs = loadDocs("./dataSet/dataDocs.pkl")

        # get all tokens from dataDocs.
        dataTokens = [token for doc in dataDocs for token in doc[0]]

        # get TOP N tokens with the highest frequency of output.
        saveTopNwords(dataTokens=dataTokens, topN=topN, filePath="./dataSet/top_%d_maxLen_%d_words.json" % (topN, maxLen))
        topN_words = loadTopNwords("./dataSet/top_%d_maxLen_%d_words.json" % (topN, maxLen))

        # convert frequency words dictionary to lookup table.
        saveTopNindex(topN_words=topN_words, topN=topN, filePath="./dataSet/top_%d_maxLen_%d_index.json" % (topN, maxLen))
        topN_index = loadTopNindex("./dataSet/top_%d_maxLen_%d_index.json" % (topN, maxLen))

    else:
        topN_words = loadTopNwords("./dataSet/top_%d_maxLen_%d_words.json" % (topN, maxLen))
        topN_index = loadTopNindex("./dataSet/top_%d_maxLen_%d_index.json" % (topN, maxLen))

    # ==================================================================================================
    # === process that encode train data[ratings_train.txt] to integer data type(lookup table index) ===
    # ==================================================================================================

    if not os.path.isfile("./dataSet/train_x_for_top_%d_maxLen_%d.npy" % (topN, maxLen)) and not os.path.isfile("./dataSet/train_y_for_top_%d_maxLen_%d.npy" % (topN, maxLen)):
        trainDocs = loadDocs("./dataSet/trainDocs.pkl")
        print("train Docs list length :", len(trainDocs))

        # divide by input x and label y
        train_x = []
        train_y = []
        for x, y in trainDocs:
            train_x.append(x)
            train_y.append(y)
        prev_train_x = train_x.copy()

        lookupTable = loadTopNindex("./dataSet/top_%d_maxLen_%d_index.json" % (topN, maxLen))
        train_x = encodeToInt(train_x, lookupTable, maxLen)  # get numpy array that converted from morpheme to interger(lookup table index value)
        train_y = np.array(train_y)

        np.save("./dataSet/train_x_for_top_%d_maxLen_%d.npy" % (topN, maxLen), train_x)
        np.save("./dataSet/train_y_for_top_%d_maxLen_%d.npy" % (topN, maxLen), train_y)
    else:
        print("train set already saved.")
        train_x = np.load("./dataSet/train_x_for_top_%d_maxLen_%d.npy" % (topN, maxLen))
        train_y = np.load("./dataSet/train_y_for_top_%d_maxLen_%d.npy" % (topN, maxLen))

    # ==================================================================================================
    # ==== process that encode test data[ratings_test.txt] to integer data type(lookup table index) ====
    # ==================================================================================================

    if not os.path.isfile("./dataSet/test_x_for_top_%d_maxLen_%d.npy" % (topN, maxLen)) and not os.path.isfile("./dataSet/test_y_for_top_%d_maxLen_%d.npy" % (topN, maxLen)):
        testDocs = loadDocs("./dataSet/testDocs.pkl")
        print("test Docs list length :", len(testDocs))

        # divide by input x and label y
        test_x = []
        test_y = []
        for x, y in testDocs:
            test_x.append(x)
            test_y.append(y)

        lookupTable = loadTopNindex("./dataSet/top_%d_maxLen_%d_index.json" % (topN, maxLen))
        test_x = encodeToInt(test_x, lookupTable, maxLen)  # get numpy array that converted from morpheme to interger(lookup table index value)
        test_y = np.array(test_y)

        np.save("./dataSet/test_x_for_top_%d_maxLen_%d.npy" % (topN, maxLen), test_x)
        np.save("./dataSet/test_y_for_top_%d_maxLen_%d.npy" % (topN, maxLen), test_y)
    else:
        print("train set already saved.")
        train_x = np.load("./dataSet/train_x_for_top_%d_maxLen_%d.npy" % (topN, maxLen))
        train_y = np.load("./dataSet/train_y_for_top_%d_maxLen_%d.npy" % (topN, maxLen))
