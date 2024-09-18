def RAUCS(conts, loops=1000000, threshold=0.15):
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

def RACS(conts, loops=1000000, threshold=0.5):
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