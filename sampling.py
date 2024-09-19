import pandas as pd
import numpy as np
import random
import math

def sample_from_bayesnet(w0, w1,w2,  n_samples):
  #w0: B->E の重み
  #w1: C->E の重み
  #w2: B->C の重み
  #n_samples: サンプルサイズ
  a = w2 * (1 - (1 - w0) * (1 - w1))# P(E=1,C=1)
  b = w2 * (1 - w0) * (1 - w1)# P(E=0,C=1)
  c = (1 - w2) * w1# P(E=1,C=0)
  d = (1 - w2) * (1 - w1)# P(E=0,C=0)
  probabilities = [a, b, c, d]
  # print(a+b+c+d)

  # 値のリスト
  values = ['A', 'B', 'C', 'D']

  # サンプリングを実行
  samples = np.random.choice(values, size=n_samples, p=probabilities)

  # 各値が得られた回数をカウント
  counts = {value: 0 for value in values}
  for sample in samples:
      counts[sample] += 1

  # カウントを配列に変換
  count_array = [counts[value] for value in values]

  return count_array

def sample_from_distribution(a, b, c, d, n_samples):
    # 確率のリスト
    probabilities = [a, b, c, d]

    # 値のリスト
    values = ['A', 'B', 'C', 'D']

    # サンプリングを実行
    samples = np.random.choice(values, size=n_samples, p=probabilities)

    # 各値が得られた回数をカウント
    counts = {value: 0 for value in values}
    for sample in samples:
        counts[sample] += 1

    # カウントを配列に変換
    count_array = [counts[value] for value in values]

    return count_array