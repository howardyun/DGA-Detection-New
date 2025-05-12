import glob
import os

from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import entropy
import numpy as np
import pandas as pd


def average_string_length(string_list):
    """
    Calculate the average length of strings in a given list of strings.

    Parameters:
    string_list (list): A list of strings.

    Returns:
    float: The average length of the strings in the list.
    """
    if not string_list:  # Check if the list is empty
        return 0
    return sum(len(s) for s in string_list) / len(string_list)


def compute_ngram_entropy(domain_list, n=2):
    """计算域名列表的N-gram信息熵"""
    # 使用CountVectorizer提取N-gram频率特征
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    X = vectorizer.fit_transform(domain_list)
    X_freq = np.array(X.sum(axis=0)).flatten() / X.sum()
    # 计算信息熵
    entropies = entropy(X_freq)
    return entropies

# Data structure for record
catgory = ['catgory']
entropies = ['entropy']
Avg_lens = ['Avg_lens']

# 良性域名列表
domain_beinign_domainList = pd.read_csv('../../data/Benign/top-1m.csv').iloc[:, 1].tolist()

# 计算并打印信息熵
entropy1 = compute_ngram_entropy(domain_beinign_domainList, n=2)
catgory.append('benign')
entropies.append(entropy1)
Avg_lens.append(average_string_length(domain_beinign_domainList))

csv_files = glob.glob(os.path.join('../../data/DGA/2016-09-19-dgarchive_full', '*.csv'))
for file in csv_files:
    domain_malicious_domainList = pd.read_csv(file).iloc[:,
                                  0].tolist()
    entropy2 = compute_ngram_entropy(domain_malicious_domainList, n=3)
    catgory.append(file.split('/')[-1].split('.')[0])
    entropies.append(entropy2)
    Avg_lens.append(average_string_length(domain_malicious_domainList))

print(f"Computed catgory: {catgory}")
print(f"Computed entropies: {entropies}")
print(f"Computed Avg_lens: {Avg_lens}")
