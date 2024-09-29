import pandas as pd
import numpy as np
import random
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import japanize_matplotlib
import importlib
import models
import sampling

importlib.reload(models)
importlib.reload(sampling)


# シミュレーションする場所
sample_sizes = [7,21,56]
df = pd.DataFrame()
for sample_size in sample_sizes:
  for i in range(5000):
    w0 = random.uniform(0,1)
    w1 = random.uniform(0,1)
    w2 = random.uniform(0,1)
    average_baserate = int((w2+w0)*5)/10
    sampled_values = sampling.sample_from_bayesnet(w0,w1,w2 ,sample_size)
    # paris = sampled_values[0] / (sampled_values[0] + sampled_values[1] + sampled_values[2])
    paris = 1
    # dp = (sampled_values[0] / (sampled_values[0] + sampled_values[1])) - (sampled_values[2] / (sampled_values[2] + sampled_values[3]))
    dp = 1
    RACS = models.J_RACS(sampled_values)
    RAUCS = models.J_RAUCS(sampled_values)
    # データフレームに追加
    df = pd.concat([df, pd.DataFrame([{'RACS':RACS,'RAUCS':RAUCS,'paris':paris,'dp':dp,'average_baserate': average_baserate,'sample_size':sample_size,'ace':w1*(1-w0)}])], ignore_index=True)


# Grouping the DataFrame by 'average_baserate' and 'sample_size'
grouped_df = df.groupby(['average_baserate', 'sample_size'])

# Creating a summary DataFrame to show the number of data points for each combination of average_baserate and sample_size
summary_df = grouped_df.size().reset_index(name='data_count')
print(summary_df)

# Initialize a dictionary to store results for each sample size
r2_results = {}

# For each group (each unique sample size), calculate the R^2 values
for (average_baserate, sample_size), group in grouped_df:
    if sample_size not in r2_results:
        r2_results[sample_size] = {'average_baserate': [], 'r2_ace_racs': [], 'r2_ace_raucs': []}
    
    X_racs = group[['RACS']]
    X_raucs = group[['RAUCS']]
    y_ace = group['ace']
    
    # Linear regression models
    model_racs = LinearRegression().fit(X_racs, y_ace)
    model_raucs = LinearRegression().fit(X_raucs, y_ace)
    
    # Predictions
    y_pred_racs = model_racs.predict(X_racs)
    y_pred_raucs = model_raucs.predict(X_raucs)
    
    # R^2 scores
    r2_ace_racs = r2_score(y_ace, y_pred_racs)
    r2_ace_raucs = r2_score(y_ace, y_pred_raucs)
    
    # Append results
    r2_results[sample_size]['average_baserate'].append(average_baserate)
    r2_results[sample_size]['r2_ace_racs'].append(r2_ace_racs)
    r2_results[sample_size]['r2_ace_raucs'].append(r2_ace_raucs)

# Plotting results for each sample size
figures = {}
for sample_size, results in r2_results.items():
    fig, ax = plt.subplots()
    ax.plot(results['average_baserate'], results['r2_ace_racs'], label='R2: ace vs RACS', marker='o')
    ax.plot(results['average_baserate'], results['r2_ace_raucs'], label='R2: ace vs RAUCS', marker='x')
    ax.set_xlabel('Average Baserate')
    ax.set_ylabel('各モデルと母集団aceとのR²')
    ax.set_title(f'Sample Size: {sample_size}')
    ax.legend()
    figures[sample_size] = fig

    # Saving each figure as a PNG image
    fig_filename = f"ace_sim_sample_size_{sample_size}.png"
    fig.savefig(fig_filename)
    print(f"Figure saved as {fig_filename}")

plt.show()