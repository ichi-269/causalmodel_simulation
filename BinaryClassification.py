import pandas as pd
import numpy as np
import random
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.special import betaln
from scipy.stats import sem, t
import importlib
import models

importlib.reload(models)

"""サンプリング関数：エッジの重みとCの生起確率を引数として刺激を作成する"""
def compare_two_values(a, b):
    return 1 if a >= b else 0

def RAUCS(conts, loops=10000, threshold=0.15):
    rng = np.random.default_rng()  # 乱数ジェネレーター
    power = np.zeros((loops, 3))

    # ループを実行して条件を満たす乱数を生成
    # 同時確率をpower1*power2 以上にする
    # power1 + power2 < 1 で条件付けする
    for i in range(loops):
        while 1:
            power1 = rng.uniform(0, threshold)
            power2 = rng.uniform(0, threshold)
            if power1 + power2 < 1:
                break
        #   if power1 + power2 < 1:
        power0 = rng.uniform(power1*power2, min(1 - power1 - power2,power1,power2))
        #   else:
        #     power0 = rng.uniform(power1 + power2 - 1, min(power1,power2))
        power[i] = [power0,power1,power2]

    a, b, c, d = conts
    # print(a, b, c, d)

    # power[:, 0] 原因と結果のw
    # power[:, 1] p(c)
    # power[:, 2] p(e)

    probs1 = [
        power[:, 0],
        power[:, 1] - power[:, 0],
        power[:, 2] - power[:, 0],
        power[:, 0] + 1 - power[:, 1] - power[:, 2],
    ]
    # print(probs1)
    probs0 = [
        power[:, 1] * power[:, 2],
        power[:, 1] - power[:, 1] * power[:, 2],
        power[:, 2] - power[:, 1] * power[:, 2],
        (power[:, 1] * power[:, 2]) + 1 - (power[:, 1] + power[:, 2]),
    ]
    # print(probs0)

    loglike1 = np.sum((np.ones((loops, 1)) * np.array(conts)) * np.log(probs1).T, axis=1)
    like1 = sum(np.exp(loglike1)) * (1/loops)

    loglike0 = np.sum((np.ones((loops, 1)) * np.array(conts)) * np.log(probs0).T, axis=1)
    like0 = sum(np.exp(loglike0)) * (1/loops)

    logscore = np.log(like1/like0)
    return [like0/(like0+like1), like1/(like0+like1)]
    # return [compare_two_values(like0, like1),compare_two_values(like1, like0)]

def RACS(conts, loops=10000, threshold=0.5):
    # 希少性仮定下のcausal support
    #背景因から原因へのエッジを追加し，P(C)を計算に用いることができるようになっている

    rng = np.random.default_rng()  # 乱数ジェネレーター
    power = np.zeros((loops, 3))

    # ループを実行して条件を満たす乱数を生成
    for i in range(loops):
      power0 = rng.uniform(0, 1)
      power1 = rng.uniform(0, threshold)
      power2 = rng.uniform(0, threshold)
      power[i] = [power0,power1,power2]

    a, b, c, d = conts
    # print(a, b, c, d)

    # power[:, 0] 原因と結果のw
    # power[:, 1] 背景と結果のw
    # power[:, 2] 背景と原因のw

    # P(C=1) = power[:, 2]

    probs1 = [
        power[:, 2] * (1 - (1 - power[:, 0]) * (1 - power[:, 1])),# P(E=1,C=1)
        power[:, 2] * (1 - power[:, 0]) * (1 - power[:, 1]),# P(E=0,C=1)
        (1 - power[:, 2]) * power[:, 1],# P(E=1,C=0)
        (1 - power[:, 2]) * (1 - power[:, 1]),# P(E=0,C=0)
    ]
    probs0 = [
        power[:, 2] * power[:, 1],# P(E=1,C=1)
        power[:, 2] * (1 - power[:, 1]),# P(E=0,C=1)
        (1 - power[:, 2]) * power[:, 1],# P(E=1,C=0)
        (1 - power[:, 2]) * (1 - power[:, 1]),# P(E=0,C=0) 1 -w1 -w2 +w1 w2
    ]

    loglike1 = np.sum((np.ones((loops, 1)) * np.array(conts)) * np.log(probs1).T, axis=1)
    like1 = sum(np.exp(loglike1)) * (1/loops)

    loglike0 = np.sum((np.ones((loops, 1)) * np.array(conts)) * np.log(probs0).T, axis=1)
    like0 = sum(np.exp(loglike0)) * (1/loops)

    logscore = np.log(like1/like0)
    return [like0/(like0+like1), like1/(like0+like1)]    
    # return [compare_two_values(like0, like1),compare_two_values(like1, like0)]

def sample_from_bayesnet(w0, w1,w2,  n_samples):
  #w0: B->E の重み
  #w1: C->E の重み
  #w2: B->C の重み
  #n_samples: サンプルサイズ
  # a = w2 * (1-(1-w0)*(1-w1))
  # b = w2 * (1 - (1-(1-w0)*(1-w1)))
  # c = (1 - w2) * (1-(1-w0))
  # d = (1 - w2) * (1 - (1-(1-w0)))
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

def sse(a, b):
    # 二乗和誤差を計算
    sse = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    return sse

def round_to_tenth(value):
    return round(value * 10) / 10

"""### シミュレーションする場所"""

sample_sizes = [7,21,56]
df = pd.DataFrame()
for sample_size in sample_sizes:
  for i in range(10000):
    processed = 0 # モデルが定義されるサンプルの数を数えるための変数
    RACS_sum_sse = 0
    RAUCS_sum_sse = 0
    if random.random() < 0.5:
      w1 = random.uniform(0,1)
    else:
      w1 = 0
    w2 = random.uniform(0,1)
    w0 = random.uniform(0,1)
    average_baserate = int((w2+w0)*5)/10
    while processed < 1:
      try:
        if w1 == 0:
          ans = [1,0]
        else:
          ans = [0,1]
        sampled_values = sample_from_bayesnet(w0,w1,w2 ,sample_size)
        RACS_val = RACS(sampled_values)
        RACS_sum_sse += sse(RACS_val, ans)
        RAUCS_val = RAUCS(sampled_values)
        RAUCS_sum_sse += sse(RAUCS_val, ans)
        # 処理が成功した場合にカウンタを増やす
        processed += 1
      except Exception as e:
        # エラーが発生した場合にエラーメッセージを表示し、処理をスキップ
        print(f"エラーが発生しました: {e}")
        continue
        # データフレームに追加
    RACS_mse = RACS_sum_sse / processed
    RAUCS_mse = RAUCS_sum_sse / processed
    df = pd.concat([df, pd.DataFrame([{'RAUCS_value':RAUCS_mse,'RACS_value':RACS_mse,'average_baserate': average_baserate,'sample_size':sample_size,'w0':w0,'w1':w1,'ace':w1*(1-w0)}])], ignore_index=True)
print(df)

grouped = df.groupby(['sample_size', 'average_baserate']).mean().reset_index()
# プロット
sample_sizes = grouped['sample_size'].unique()
for sample_size in sample_sizes:
    subset = grouped[grouped['sample_size'] == sample_size]
    plt.figure(figsize=(10, 6))
    plt.plot(subset['average_baserate'], subset['ace'], label='ace')
    plt.xlabel('average_baserate')
    plt.ylabel('Average Value')
    plt.title(f'Sample Size: {sample_size}')
    plt.legend()
    plt.grid(True)
    plt.show()

grouped = df.groupby(['sample_size', 'average_baserate']).mean().reset_index()

# プロット
sample_sizes = grouped['sample_size'].unique()
for sample_size in sample_sizes:
    subset = grouped[grouped['sample_size'] == sample_size]
    plt.figure(figsize=(10, 6))
    plt.plot(subset['average_baserate'], subset['RAUCS_value'], label='RAUCS')
    plt.plot(subset['average_baserate'], subset['RACS_value'], label='RACS')
    plt.xlabel('P(C), P(E)')
    plt.ylabel('mse')
    plt.title(f'Sample Size: {sample_size}')
    plt.legend()
    plt.grid(True)
    plt.show()