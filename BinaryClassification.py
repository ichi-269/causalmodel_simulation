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
import sampling

importlib.reload(models)
importlib.reload(sampling)

#ベイズネットからサンプリングを行い，それぞれのモデルの二値分類性能を比較する

def compare_two_values(a, b):
    return 1 if a >= b else 0

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
  for i in range(1000):
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
        sampled_values = sampling.sample_from_bayesnet(w0,w1,w2 ,sample_size)
        RACS_val = models.BC_J_RACS(sampled_values)
        RACS_sum_sse += sse(RACS_val, ans)
        RAUCS_val = models.BC_J_RAUCS(sampled_values)
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
  plt.xlabel('average_baserate')
  plt.ylabel('mse')
  plt.title(f'Sample Size: {sample_size}')
  plt.legend()
  plt.grid(True)
  # Saving each figure as a PNG image
  fig_filename = f"BC_simsample_size_{sample_size}.png"
  plt.savefig(fig_filename)
  print(f"Figure saved as {fig_filename}")
  plt.show()