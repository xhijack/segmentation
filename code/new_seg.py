import sys

reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd
import tools
import numpy as np
import uts
from gensim.models import Word2Vec
from nltk import tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()
from nltk.metrics import windowdiff, pk


STDEVIATION =0.0009

def c99(sent_tokenized, window=5, K=10):
    model = uts.C99(window=window)
    boundary = model.segment([" ".join(s) for s in sent_tokenized])
    return "".join([str(i) for i in boundary])


def text_tiling(sent_tokenized, window=5, K=10):
    model = uts.TextTiling(window=window)
    boundary = model.segment([" ".join(s) for s in sent_tokenized])
    hypt = "".join([str(i) for i in boundary])
    return hypt


def load_stop_words(stopword='stopword.txt'):
    if stopword:
        file = open(stopword, 'r')
        text = file.read()
        return set(text.split("\n"))


def load_file_txt(source):
    file = open(source, 'r')
    text = file.read()
    return text.split("\n")


def load_data(name='id.albaqarah.cut.txt'):
    data = load_file_txt(name)
    expected = "".join([str(i.split(",")[0]) for i in data])
    return data, expected


def stem(word):
    return stemmer.stem(word)


def gensig_model(X, minlength=1, maxlength=None, lam=0.0):
    N, D = X.shape
    over_sqrtD = 1. / np.sqrt(D)
    cs = np.cumsum(X, 0)
    # print("SQRT:", over_sqrtD)

    def sigma(a, b):
        length = (b - a)
        if minlength:
            if length < minlength: return np.inf
        if maxlength:
            if length > maxlength: return np.inf

        tot = cs[b - 1].copy()
        if a > 0:
            tot -= cs[a - 1]
        signs = np.sign(tot)
        # print("A: {} B: {}, Nilai sigma: {}".format(a,b, -over_sqrtD * (signs*tot).sum()))
        # print("sigma (a b):",a,b,-over_sqrtD * (signs*tot).sum())
        hade = tot.sum()
        return -over_sqrtD * (signs * tot).sum(), hade

    return sigma


def greedysplit(n, k, sigma):
    """ Do a greedy split """
    splits = [n]
    s = sigma(0, n)

    def score(splits, sigma):
        splits = sorted(splits)

        result = []
        # result = sum( sigma(a,b) for (a,b) in tools.seg_iter(splits) )

        check = []
        result2 = []
        for (a, b) in tools.seg_iter(splits):
            check.append([a, b])
            o, n = sigma(a, b)
            result.append(o)
            result2.append(n)

        # print(result)
        # print splits, check, sum(result)
        # print(result2, sum(result2))
        # print("--")

        return sum(result)

    new_score = []
    k = k - 1
    while k > 0:
        usedinds = set(splits)
        new_arr = []
        # print "menghitung ke K-", k
        # print("--begin scoring----")
        for i in xrange(1, n):
            if i not in usedinds:
                new_arr.append([score(splits + [i], sigma), splits + [i]])
        # print("--end scoring----")
        #
        # print("pemilihan batas:", min(new_arr))
        # print("sorted batas:", sorted(min(new_arr)[1]))
        new = min(new_arr)
        print(new_arr)
        new_score.append(new)
        splits = new[1]
        s = new[0]
        k -= 1

    if 1 not in splits:
        splits = splits + [1]

    return sorted(splits), new_arr


def load_model(model_name):
    model = Word2Vec.load(model_name)
    index2word_set = set(model.wv.index2word)
    return model, index2word_set


def avg_feature_vector(words, model, num_features, index2word_set):
    feature_vec = np.zeros((num_features,), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec


def word2sent(model, sent_tokenized, index2word_set):
    X = []
    for i in range(0, len(sent_tokenized)):
        sent1avg = avg_feature_vector(sent_tokenized[i], model, 400, index2word_set)
        X.append(sent1avg)
    X = np.array(X)
    return X



def OriginalGreedy(sent_tokenized, window=5, K=10):
    X = word2sent(_model, sent_tokenized, index2word_set)
    sig = gensig_model(X)
    spl, e = greedysplit(X.shape[0], K, sig)

    segs = [1] * len(sent_tokenized)
    for i in range(1, len(sent_tokenized)):
        if i in spl:
            segs[i] = 1
        else:
            segs[i] = 0
    segs[1] = 0
    return "".join([str(i) for i in segs])


models = ["models/w2vec_wiki_id_non_case"]


def NewSegV1(sent_tokenized, window=5, K=10):
    # _model, index2word_set = load_model(model)
    X = word2sent(_model, sent_tokenized, index2word_set)

    results = []
    i = 0
    X_ = len(X)
    cursor = window
    splits = [1]
    while cursor - window < X_ and X[i:cursor].shape[0] > 1:
        K = 2
        sig = gensig_model(X[i:cursor])
        # print("window:", window)
        print("trying segment from {} to {}".format(i, cursor))
        # if X[i:cursor].shape[0] == 1:
        #     import pdb
        #     pdb.set_trace()
        spl, e = greedysplit(X[i:cursor].shape[0], K, sig)

        stdeviation = np.std([a[0] for a in e])
        print("cut or no:", np.std([a[0] for a in e]), splits, spl)
        if stdeviation > STDEVIATION:  #0.0206:
            i = spl[1]
            if len(splits) == 1:
                new_seg1 = i
            else:
                new_seg1 = splits[-1] + i
            splits.append(new_seg1)
            i = new_seg1
            cursor = i + window
        else:
            cursor = cursor + window
        # elif stdeviation == 0:
        #     i = X_ + 1
        # else:
        #     cursor = i + (2 * window)
        #     if cursor - window == X_ - 1:
        #         break
        #     if cursor > X_:
        #         cursor = X_
        #     if i + window > X_:
        #         break

    # splits, e = new_greedy_split(X.shape[0], K, sig, 10)
    # print(splits)

    # sim = 1 - spatial.distance.cosine(X[0],  X[1])

    # exp = "".join([sent.split(",")[0] for sent in sents])
    # pdb.set_trace()

    segs = [1] * len(sent_tokenized)
    for i in range(1, len(sent_tokenized)):
        if i in splits:
            segs[i] = 1
        else:
            segs[i] = 0
    # print("expected  ", expected)
    segs[1] = 0
    # print("experiment", "".join([str(i) for i in segs]))
    # print(windowdiff(expected, "".join([str(i) for i in segs]), K))
    # results.append(windowdiff(expected, "".join([str(i) for i in segs]), K))

    # print(results)
    return "".join([str(i) for i in segs])


def NewSeg(sent_tokenized, window=5, K=10):
    # _model, index2word_set = load_model(model)
    X = word2sent(_model, sent_tokenized, index2word_set)

    results = []
    i = 0
    X_ = len(X)
    cursor = window
    splits = [1]
    while cursor < X_:
        K = 2
        sig = gensig_model(X[i:cursor])
        print("window:", window)
        print("trying segment from {} to {}".format(i, cursor))
        spl, e = greedysplit(X[i:cursor].shape[0], K, sig)
        i = spl[1]
        if len(splits) == 1:
            new_seg1 = i
        else:
            new_seg1 = splits[-1] + i
        splits.append(new_seg1)
        i = new_seg1
        cursor = i + window

    # splits, e = new_greedy_split(X.shape[0], K, sig, 10)
    # print(splits)

    # sim = 1 - spatial.distance.cosine(X[0],  X[1])

    # exp = "".join([sent.split(",")[0] for sent in sents])
    # pdb.set_trace()

    segs = [1] * len(sent_tokenized)
    for i in range(1, len(sent_tokenized)):
        if i in splits:
            segs[i] = 1
        else:
            segs[i] = 0
    # print("expected  ", expected)
    segs[1] = 0
    # print("experiment", "".join([str(i) for i in segs]))
    # print(windowdiff(expected, "".join([str(i) for i in segs]), K))
    # results.append(windowdiff(expected, "".join([str(i) for i in segs]), K))

    # print(results)
    return "".join([str(i) for i in segs])



import argparse

if __name__ == "__main__":
    K = 8
    isStopWord = [True, False]
    isStemmed = [True, False]

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="please input the model")
    args = parser.parse_args()
    model = "models/{}".format(args.model)
    _model, index2word_set = load_model(model)

    data = [
        ['data/id.albaqarah.original.txt', 55],
        ['data/juz_amma.txt', 60],
        ['data/al-imron.txt', 34],
        ['data/annisa.txt', 33],
        ['data/almaaidah.txt', 22],
        ['data/alanam.txt', 13],
        # ['data/fadhail-amal.txt', 6],
        # ['data/sintesis.detik.txt', 10],
        # ['data/id.albaqarah.cut.txt', 3],
        # ['data/sintesis.extreme1.txt', 6],
        # ['data/sintesis.extreme2.txt', 5],
        # ['data/sintesis.extreme3.txt', 8],
        # ['data/sintesis.extreme4.txt', 8],
        # ['data/sintesis.extreme5.txt', 4],
        # ['data/sintesis.extreme6.txt', 9],
        # ['data/sintesis.extreme7.txt', 9],
        # ['data/sintesis.extreme8.txt', 7],
        # ['data/sintesis.extreme9.txt', 9],
        # ['data/sintesis.extreme10.txt', 9],
        #
        # ['data/sintesis.extreme11.txt', 11],
        # ['data/sintesis.extreme12.txt', 9],
        # ['data/sintesis.extreme13.txt', 8],
        # ['data/sintesis.extreme14.txt', 9],
        # ['data/sintesis.extreme15.txt', 7],
        # ['data/sintesis.extreme16.txt', 7],
        # ['data/sintesis.extreme17.txt', 8],
        # ['data/sintesis.extreme18.txt', 8],
        # ['data/sintesis.extreme19.txt', 8],
        # ['data/sintesis.extreme20.txt', 7],
        #
        # ['data/sintesis.extreme21.txt', 8],
        # ['data/sintesis.extreme22.txt', 8],
        # ['data/sintesis.extreme23.txt', 8],
        # ['data/sintesis.extreme24.txt', 9],
        # ['data/sintesis.extreme25.txt', 10],
        # ['data/sintesis.extreme26.txt', 10],
        # ['data/sintesis.extreme27.txt', 9],
        # ['data/sintesis.extreme28.txt', 8],
        # ['data/sintesis.extreme29.txt', 9],
        # ['data/sintesis.extreme30.txt', 8],

        # ['data/sintesis.kompas.android.smooth.txt', 7],
        # ['data/sintesis.kompas.android.smooth2.txt', 3],
        # ['data/sintesis.kompas.politik.smooth.txt', 3],
        # ['data/sintesis.kompas.tekno.middle.txt', 7],
        # ['data/sintetis.konsyar.sholat.smooth.txt', 5]
    ]

    # sents, expected = get_albaqarah('id.albaqarah.original.v2.txt')
    stopword = load_stop_words()

    methods = [NewSegV1, OriginalGreedy, c99, text_tiling]
    # import pdb
    # pdb.set_trace()
    results = []
    for sw in isStopWord:
        for ist in isStemmed:
            # if sw and ist: #check all true. please remove after finished
            for window in [10, 15, 20]:

                for expe in data:
                    sents, expected = load_data(expe[0])
                    sent_tokenized = []
                    for sent in sents:
                        words = tokenize.word_tokenize(sent)
                        if sw:
                            if ist:
                                sent_tokenized.append(
                                    [stem(word.lower()) for word in words if
                                     word.lower() not in stopword and word.isalpha()])
                                # for word in words:
                                #     if word.lower() not in stopword and word.isalpha():
                                #         sent_tokenized.append(stem(word.lower()))

                            else:
                                sent_tokenized.append(
                                    [word.lower() for word in words if word.lower() not in stopword and word.isalpha()])
                        else:
                            if ist:
                                sent_tokenized.append(
                                    [stem(word.lower()) for word in words if word.isalpha()])
                            else:
                                sent_tokenized.append(
                                    [word.lower() for word in words if word.isalpha()])

                    # tt = TextTilingTokenizer(demo_mode=False, stopwords=sw, k=56, w=20)
                    # s, ss, d, b = tt.tokenize([" ".join(sent) for sent in sent_tokenized])

                    for method in methods:
                        # try:
                        result = method(sent_tokenized, window, expe[1])
                        diff = windowdiff(expected, result, expe[1])
                        pk_diff = pk(expected, result, expe[1])
                        # print(result, expe[1])

                        record = {
                            'File': expe[0],
                            'Method Name': method.__name__,
                            'window': window,
                            'hypt segment': result,
                            'expe segment': expected,
                            'window diff': diff,
                            'isStemmed': ist,
                            'isStopped': sw
                        }
                        results.append(record)
                        print(
                            "Method {} | File: {} | StopWord: {} | Stemmed: {} | result: {} | {} {}".format(
                                method.__name__, expe,
                                sw, ist, diff, expected, result))
                        # except:
                        #     import pdb

                        # pdb.set_trace()
                    print("===")

    df = pd.DataFrame(results)
    print(df.to_string())
    df.to_csv('df.csv')
    dfpivot = df.pivot_table(index=['File','window', 'isStemmed', 'isStopped'], columns='Method Name', values='window diff', aggfunc=np.average)
    dfpivot.to_csv('dfpivot.csv')
    print(dfpivot.to_string())
    print(df.groupby(['Method Name', 'window', 'isStemmed', 'isStopped']).mean())
    # dfgroup = df.groupby(['Method Name', 'window']).mean().
