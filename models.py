import pandas as pd
import numpy as np
import random
import math

J_RAUCS_threshold = 0.125
J_RACS_threshold = 0.5
loop_count = 100000

#稀少性仮定，同時確率　UCS
def J_RAUCS(conts, loops=loop_count, threshold=J_RAUCS_threshold):
    rng = np.random.default_rng()  # 乱数ジェネレーター
    power = np.zeros((loops, 3))

    # ループを実行して条件を満たす乱数を生成
    for i in range(loops):
      power1 = rng.uniform(0, threshold)
      power2 = rng.uniform(0, threshold)
      if power1 + power2 < 1:
        power0 = rng.uniform(0, min(power1,power2))
      else:
        power0 = rng.uniform(power1 + power2 - 1, min(power1,power2))
      power[i] = [power0,power1,power2]

    a, b, c, d = conts
    # power[:, 0] 原因と結果のw
    # power[:, 1] p(c)
    # power[:, 2] p(e)

    probs1 = [
        power[:, 0],# P(E=1|C=1)
        power[:, 1] - power[:, 0],# P(E=0|C=1)
        power[:, 2] - power[:, 0],# P(E=1|C=0)
        power[:, 0] + 1 - power[:, 1] - power[:, 2],# P(E=0|C=0)
    ]
    probs0 = [
        power[:, 1] * power[:, 2],# P(E=1|C=1)
        power[:, 1] - power[:, 1] * power[:, 2],# P(E=0|C=1)
        power[:, 2] - power[:, 1] * power[:, 2],# P(E=1|C=0)
        (power[:, 1] * power[:, 2]) + 1 - (power[:, 1] + power[:, 2]),# P(E=0|C=0)
    ]

    loglike1 = np.sum((np.ones((loops, 1)) * np.array(conts)) * np.log(probs1).T, axis=1)
    like1 = sum(np.exp(loglike1)) * (1/loops)

    loglike0 = np.sum((np.ones((loops, 1)) * np.array(conts)) * np.log(probs0).T, axis=1)
    like0 = sum(np.exp(loglike0)) * (1/loops)

    logscore = np.log(like1/like0)
    return logscore

# 希少性仮定，同時確率 CS
def J_RACS(conts, loops=loop_count, threshold=J_RACS_threshold):
    rng = np.random.default_rng()  # 乱数ジェネレーター
    power = np.zeros((loops, 3))

    # ループを実行して条件を満たす乱数を生成
    for i in range(loops):
      power0 = rng.uniform(0, 1)
      power1 = rng.uniform(0, threshold)
      power2 = rng.uniform(0, threshold)
      power[i] = [power0,power1,power2]

    a, b, c, d = conts

    # power[:, 0] 原因と結果のw
    # power[:, 1] 背景と結果のw
    # power[:, 2] 背景と原因のw

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
    return logscore

#二値分類　稀少性仮定，同時確率　UCS
def BC_J_RAUCS(conts, loops=loop_count, threshold=J_RAUCS_threshold):
    rng = np.random.default_rng()  # 乱数ジェネレーター
    power = np.zeros((loops, 3))

    # ループを実行して条件を満たす乱数を生成
    for i in range(loops):
      power1 = rng.uniform(0, threshold)
      power2 = rng.uniform(0, threshold)
      if power1 + power2 < 1:
        power0 = rng.uniform(0, min(power1,power2))
      else:
        power0 = rng.uniform(power1 + power2 - 1, min(power1,power2))
      power[i] = [power0,power1,power2]

    a, b, c, d = conts
    # power[:, 0] 原因と結果のw
    # power[:, 1] p(c)
    # power[:, 2] p(e)

    probs1 = [
        power[:, 0],# P(E=1|C=1)
        power[:, 1] - power[:, 0],# P(E=0|C=1)
        power[:, 2] - power[:, 0],# P(E=1|C=0)
        power[:, 0] + 1 - power[:, 1] - power[:, 2],# P(E=0|C=0)
    ]
    probs0 = [
        power[:, 1] * power[:, 2],# P(E=1|C=1)
        power[:, 1] - power[:, 1] * power[:, 2],# P(E=0|C=1)
        power[:, 2] - power[:, 1] * power[:, 2],# P(E=1|C=0)
        (power[:, 1] * power[:, 2]) + 1 - (power[:, 1] + power[:, 2]),# P(E=0|C=0)
    ]

    loglike1 = np.sum((np.ones((loops, 1)) * np.array(conts)) * np.log(probs1).T, axis=1)
    like1 = sum(np.exp(loglike1)) * (1/loops)

    loglike0 = np.sum((np.ones((loops, 1)) * np.array(conts)) * np.log(probs0).T, axis=1)
    like0 = sum(np.exp(loglike0)) * (1/loops)

    logscore = np.log(like1/like0)
    return [like0/(like0+like1), like1/(like0+like1)]

# 二値分類　希少性仮定，同時確率 CS
def BC_J_RACS(conts, loops=loop_count, threshold=J_RACS_threshold):
    rng = np.random.default_rng()  # 乱数ジェネレーター
    power = np.zeros((loops, 3))

    # ループを実行して条件を満たす乱数を生成
    for i in range(loops):
      power0 = rng.uniform(0, 1)
      power1 = rng.uniform(0, threshold)
      power2 = rng.uniform(0, threshold)
      power[i] = [power0,power1,power2]

    a, b, c, d = conts

    # power[:, 0] 原因と結果のw
    # power[:, 1] 背景と結果のw
    # power[:, 2] 背景と原因のw

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