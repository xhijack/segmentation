import sys

from nltk import FreqDist

reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd
import nltk.tokenize

def load_file_txt(source):
    file = open(source, 'r')
    text = file.read()
    return text.split("\n")

def load_data(name='id.albaqarah.cut.txt'):
    data = load_file_txt(name)
    expected = "".join([str(i.split(",")[0]) for i in data])
    return data, expected

def load_datas():
    return [
        # ['data/id.albaqarah.original.txt', 55],
        # ['data/al-imron.txt', 34],
        # ['data/annisa.txt', 33],
        # ['data/almaaidah.txt', 22],
        # ['data/alanam.txt', 13],
        # ['data/juz_amma.txt', 60],

        ['data/sintesis.extreme1.txt', 5],
        ['data/sintesis.extreme2.txt', 6],
        ['data/sintesis.extreme3.txt', 8],
        ['data/sintesis.extreme4.txt', 8],
        ['data/sintesis.extreme5.txt', 4],
        ['data/sintesis.extreme6.txt', 9],
        ['data/sintesis.extreme7.txt', 9],
        ['data/sintesis.extreme8.txt', 7],
        ['data/sintesis.extreme9.txt', 9],
        ['data/sintesis.extreme10.txt', 9],

        ['data/sintesis.extreme11.txt', 11],
        ['data/sintesis.extreme12.txt', 9],
        ['data/sintesis.extreme13.txt', 8],
        ['data/sintesis.extreme14.txt', 9],
        ['data/sintesis.extreme15.txt', 7],
        ['data/sintesis.extreme16.txt', 6],
        ['data/sintesis.extreme17.txt', 7],
        ['data/sintesis.extreme18.txt', 8],
        ['data/sintesis.extreme19.txt', 8],
        ['data/sintesis.extreme20.txt', 7],
        #
        ['data/sintesis.extreme21.txt', 8],
        ['data/sintesis.extreme22.txt', 8],
        ['data/sintesis.extreme23.txt', 8],
        ['data/sintesis.extreme24.txt', 9],
        ['data/sintesis.extreme25.txt', 10],
        ['data/sintesis.extreme26.txt', 10],
        ['data/sintesis.extreme27.txt', 9],
        ['data/sintesis.extreme28.txt', 8],
        ['data/sintesis.extreme29.txt', 9],
        ['data/sintesis.extreme30.txt', 8],
        #
        # ['data/sintesis.kompas.android.smooth.txt', 7],
        # ['data/sintesis.kompas.android.smooth2.txt', 3],
        # ['data/sintesis.kompas.politik.smooth.txt', 3],
        # ['data/sintesis.kompas.tekno.middle.txt', 7],
        # ['data/sintetis.konsyar.sholat.smooth.txt', 5]
    ]

def load_stop_words(stopword='stopword.txt'):
    if stopword:
        file = open(stopword, 'r')
        text = file.read()
        return set(text.split("\n"))

if __name__ == '__main__':
    datas = load_datas()
    stop_words = load_stop_words()
    results = []
    for data in datas:
        sent, expected = load_data(data[0])
        words = nltk.word_tokenize(" ".join(sent))


        # import pdb
        # pdb.set_trace()
        # try:
        #     tokenized_sentences = nltk.sent_tokenize(sent)
        # except:
        #     import pdb
        #     pdb.set_trace()
        #
        # print(len(tokenized_sentences))
        result = {
            'Lines': len(sent),
            'Segments': data[1],
            'Stop Words': len([w.lower() for w in words if w.lower() in stop_words and w.isalpha()]),
            'Words': len([w.lower() for w in words if w.isalpha()]),
            'Vocabulary': len(set([w.lower() for w in words if w.isalpha()]))
        }
        result1 = {
            'Word Frequncies': FreqDist(words)
        }
        print(result1)
        results.append(result)
    df = pd.DataFrame(results)
    df.to_csv('summary_quran.csv')
    print(df.to_string())