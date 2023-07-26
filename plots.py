
import csv
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import seaborn as sns

epochs = 3
batches = 46

# Create the keys projections dictionary
print("Uploading key projections")
keys1 = {
    f'epoch{epoch}': {
        f'batch{batch}':
            {} for batch in range(batches)}
    for epoch in range(epochs)}
# Fill it
for e in range(epochs):
    for b in tqdm(range(batches)):
        p = pd.read_excel('projections/key_b'+str(b)+'_e'+str(e)+'.xlsx', header=0)    # projections / projections3
        keys1['epoch'+str(e)]['batch'+str(b)] = p

# Create the queries projections dictionary
print("Uploading query projections")
queries1 = {
    f'epoch{epoch}': {
        f'batch{batch}':
            {} for batch in range(batches)}
    for epoch in range(epochs)}
# Fill it
for e in range(epochs):
    for b in tqdm(range(batches)):
        p = pd.read_excel('projections/query_b'+str(b)+'_e'+str(e)+'.xlsx', header=0)    # projections / projections3
        queries1['epoch'+str(e)]['batch'+str(b)] = p

# heatmaps
print("Saving heat-maps")
for e in range(epochs):
    for b in tqdm(range(batches)):
        p = keys['epoch'+str(e)]['batch'+str(b)]
        ax = sns.heatmap(p, cmap='jet')
        ax.set(xlabel="to", ylabel="from")
        x_ticks = np.arange(0, p.shape[1], 100)
        y_ticks = np.arange(0, p.shape[0], 100)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, rotation=90)
        ax.xaxis.tick_top()
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks)
        ax.tick_params(axis='both', which='both', labelsize=8)
        plt.title("Key Projection Weights, Epoch "+str(e+1)+" Batch "+str(b+1))
        save_path = "keys_3/e"+str(e+1)+'b'+str(b+1)+".jpg"
        plt.savefig(save_path, format="jpeg", dpi=300)
        # Remove the existing colorbar to reset it for the next iteration
        if (e>0) and (b>0):
            ax.collections[0].colorbar.remove()
        plt.clf()  # Clear the current figure to prepare for the next iteration


for e in range(epochs):
    for b in tqdm(range(batches)):
        p = queries['epoch'+str(e)]['batch'+str(b)]
        ax = sns.heatmap(p, cmap='jet')
        ax.set(xlabel="to", ylabel="from")
        x_ticks = np.arange(0, p.shape[1], 100)
        y_ticks = np.arange(0, p.shape[0], 100)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, rotation=90)
        ax.xaxis.tick_top()
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks)
        ax.tick_params(axis='both', which='both', labelsize=8)
        plt.title("Query Projection Weights, Epoch "+str(e+1)+" Batch "+str(b+1))
        save_path = "Query_3/e"+str(e+1)+'b'+str(b+1)+".jpg"
        plt.savefig(save_path, format="jpeg", dpi=300)
        # Remove the existing colorbar to reset it for the next iteration
        if (e>0) and (b>0):
            ax.collections[0].colorbar.remove()
        plt.clf()  # Clear the current figure to prepare for the next iteration

# Key Gradients
for b in tqdm(range(batches-1)):
    o = keys['epoch'+str(0)]['batch'+str(b)]
    n = keys['epoch'+str(0)]['batch'+str(b+1)]
    p = pd.DataFrame(n-o)
    ax = sns.heatmap(p, cmap='jet')
    ax.set(xlabel="to", ylabel="from")
    x_ticks = np.arange(0, p.shape[1], 100)
    y_ticks = np.arange(0, p.shape[0], 100)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, rotation=90)
    ax.xaxis.tick_top()
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    ax.tick_params(axis='both', which='both', labelsize=8)
    plt.title("Key Projection Weights Deltas, Epoch "+str(1)+" Batch "+str(b+1)+" to "+str(b+2))
    save_path = "key_grad_3/e"+str(1)+'b'+str(b+1)+".jpg"
    plt.savefig(save_path, format="jpeg", dpi=300)
    # Remove the existing colorbar to reset it for the next iteration
    if b>0:
        ax.collections[0].colorbar.remove()
    plt.clf()  # Clear the current figure to prepare for the next iteration

for b in tqdm(range(batches-1)):
    o = keys['epoch'+str(1)]['batch'+str(b)]
    n = keys['epoch'+str(1)]['batch'+str(b+1)]
    p = pd.DataFrame(n-o)
    ax = sns.heatmap(p, cmap='jet')
    ax.set(xlabel="to", ylabel="from")
    x_ticks = np.arange(0, p.shape[1], 100)
    y_ticks = np.arange(0, p.shape[0], 100)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, rotation=90)
    ax.xaxis.tick_top()
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    ax.tick_params(axis='both', which='both', labelsize=8)
    plt.title("Key Projection Weights Deltas, Epoch "+str(2)+" Batch "+str(b+1)+" to "+str(b+2))
    save_path = "key_grad_3/e"+str(2)+'b'+str(b+1)+".jpg"
    plt.savefig(save_path, format="jpeg", dpi=300)
    # Remove the existing colorbar to reset it for the next iteration
    if b>0:
        ax.collections[0].colorbar.remove()
    plt.clf()  # Clear the current figure to prepare for the next iteration

for b in tqdm(range(batches-1)):
    o = keys['epoch'+str(2)]['batch'+str(b)]
    n = keys['epoch'+str(2)]['batch'+str(b+1)]
    p = pd.DataFrame(n-o)
    ax = sns.heatmap(p, cmap='jet')
    ax.set(xlabel="to", ylabel="from")
    x_ticks = np.arange(0, p.shape[1], 100)
    y_ticks = np.arange(0, p.shape[0], 100)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, rotation=90)
    ax.xaxis.tick_top()
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    ax.tick_params(axis='both', which='both', labelsize=8)
    plt.title("Key Projection Weights Deltas, Epoch "+str(3)+" Batch "+str(b+1)+" to "+str(b+2))
    save_path = "key_grad_3/e"+str(3)+'b'+str(b+1)+".jpg"
    plt.savefig(save_path, format="jpeg", dpi=300)
    # Remove the existing colorbar to reset it for the next iteration
    if b>0:
        ax.collections[0].colorbar.remove()
    plt.clf()  # Clear the current figure to prepare for the next iteration

#------------------------------ Query Gradients ---------------------------------#
for b in tqdm(range(batches-1)):
    o = queries['epoch'+str(0)]['batch'+str(b)]
    n = queries['epoch'+str(0)]['batch'+str(b+1)]
    p = pd.DataFrame(n-o)
    ax = sns.heatmap(p, cmap='jet')
    ax.set(xlabel="to", ylabel="from")
    x_ticks = np.arange(0, p.shape[1], 100)
    y_ticks = np.arange(0, p.shape[0], 100)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, rotation=90)
    ax.xaxis.tick_top()
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    ax.tick_params(axis='both', which='both', labelsize=8)
    plt.title("Query Projection Weights Deltas, Epoch "+str(1)+" Batch "+str(b+1)+" to "+str(b+2))
    save_path = "query_grad_3/e"+str(1)+'b'+str(b+1)+".jpg"
    plt.savefig(save_path, format="jpeg", dpi=300)
    # Remove the existing colorbar to reset it for the next iteration
    if b>0:
        ax.collections[0].colorbar.remove()
    plt.clf()  # Clear the current figure to prepare for the next iteration

for b in tqdm(range(batches-1)):
    o = queries['epoch'+str(1)]['batch'+str(b)]
    n = queries['epoch'+str(1)]['batch'+str(b+1)]
    p = pd.DataFrame(n-o)
    ax = sns.heatmap(p, cmap='jet')
    ax.set(xlabel="to", ylabel="from")
    x_ticks = np.arange(0, p.shape[1], 100)
    y_ticks = np.arange(0, p.shape[0], 100)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, rotation=90)
    ax.xaxis.tick_top()
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    ax.tick_params(axis='both', which='both', labelsize=8)
    plt.title("Query Projection Weights Deltas, Epoch "+str(2)+" Batch "+str(b+1)+" to "+str(b+2))
    save_path = "query_grad_3/e"+str(2)+'b'+str(b+1)+".jpg"
    plt.savefig(save_path, format="jpeg", dpi=300)
    # Remove the existing colorbar to reset it for the next iteration
    if b>0:
        ax.collections[0].colorbar.remove()
    plt.clf()  # Clear the current figure to prepare for the next iteration

for b in tqdm(range(batches-1)):
    o = queries['epoch'+str(2)]['batch'+str(b)]
    n = queries['epoch'+str(2)]['batch'+str(b+1)]
    p = pd.DataFrame(n-o)
    ax = sns.heatmap(p, cmap='jet')
    ax.set(xlabel="to", ylabel="from")
    x_ticks = np.arange(0, p.shape[1], 100)
    y_ticks = np.arange(0, p.shape[0], 100)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, rotation=90)
    ax.xaxis.tick_top()
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    ax.tick_params(axis='both', which='both', labelsize=8)
    plt.title("Query Projection Weights Deltas, Epoch "+str(3)+" Batch "+str(b+1)+" to "+str(b+2))
    save_path = "query_grad_3/e"+str(3)+'b'+str(b+1)+".jpg"
    plt.savefig(save_path, format="jpeg", dpi=300)
    # Remove the existing colorbar to reset it for the next iteration
    if b>0:
        ax.collections[0].colorbar.remove()
    plt.clf()  # Clear the current figure to prepare for the next iteration

# ------------------- Recostructions -------------------#

reals = pd.read_excel('reconstructions/reals.xlsx', header=0)
kovas = pd.read_excel('reconstructions/kovas.xlsx', header=0)
errors = pd.read_excel('reconstructions/errors.xlsx', header=0)

# -------------- VS Input ------------#
epoch=1
batch=1
for i in tqdm(range(138)):
    real = pd.Series(reals.iloc[i*5,:])
    kova = pd.Series(kovas.iloc[i*5,:])

    # move to 0
    real_mean = real.mean()
    real = real - real_mean
    kova = kova - real_mean

    batch = batch + 1
    if batch>46:
        epoch = epoch + 1
        batch = 1

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the lines from Series 1 and Series 2
    ax.plot(real.index, real.values, label='Real', color='blue')
    ax.plot(kova.index, kova.values, label='Prediction', color='green')

    # Add labels, title, and legend
    ax.set_xlabel('series time stamp')
    ax.set_ylabel('1st feature value')
    ax.set_ylim(-1.5, 1.5)  # This will set the Y-axis limits from 0 to 6
    plt.title("Input VS Reconstruction, Epoch "+str(epoch)+" Batch "+str(batch))
    plt.legend(loc='upper right')

    save_path = "Reconstruction VS Input/e"+str(epoch)+'b'+str(batch)+".jpg"
    plt.savefig(save_path, format="jpeg", dpi=300)

# -------------- error ------------#
epoch=1
batch=0

for i in range(138):

    batch = batch + 1
    if batch>46:
        epoch = epoch + 1
        batch = 1

    mean_e = np.array(errors.iloc[i*5:i*5+5,:].abs().mean())
    std_e = np.array(errors.iloc[i*5:i*5+5,:].abs().std())

    # Calculate the 95% confidence interval
    lower_bound = mean_e - 1.96 * std_e
    upper_bound = mean_e + 1.96 * std_e

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the mean line
    ax.plot(np.arange(100), mean_e.ravel(), label='Mean', color='blue')

    # Fill the area within the confidence interval
    ax.fill_between(np.arange(100), lower_bound.ravel(), upper_bound.ravel(), color='lightblue', alpha=0.25,
                    label='95% CI')

    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Absolute-Error')
    ax.set_ylim(0, 1.5)
    plt.title('Absolute Mean Reconstruction Error, Epoch '+str(epoch)+' Batch '+str(batch))
    plt.legend(loc='upper right')
    save_path = "error/e"+str(epoch)+'b'+str(batch)+".jpg"
    plt.savefig(save_path, format="jpeg", dpi=300)



epoch=1
batch=1
for i in tqdm(range(138)):
    error

    # move to 0
    real_mean = real.mean()
    real = real - real_mean
    kova = kova - real_mean

    batch = batch + 1
    if batch>46:
        epoch = epoch + 1
        batch = 1

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the lines from Series 1 and Series 2
    ax.plot(real.index, real.values, label='Real', color='blue')
    ax.plot(kova.index, kova.values, label='Prediction', color='green')

    # Add labels, title, and legend
    ax.set_xlabel('series time stamp')
    ax.set_ylabel('1st feature value')
    ax.set_ylim(-1.5, 1.5)  # This will set the Y-axis limits from 0 to 6
    plt.title("Input VS Reconstruction, Epoch "+str(epoch)+" Batch "+str(batch))
    plt.legend(loc='upper right')

    save_path = "Reconstruction VS Input/e"+str(epoch)+'b'+str(batch)+".jpg"
    plt.savefig(save_path, format="jpeg", dpi=300)

#-------- key besides query ---------#

print("key vs query")
for e in range(epochs):
    for b in tqdm(range(batches)):

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        p = keys['epoch'+str(e)]['batch'+str(b)]
        ax0 = sns.heatmap(p, cmap='jet', ax=axes[0])
        axes[0].set_title('Key', fontsize=16)
        x_ticks = np.arange(0, p.shape[1], 100)
        y_ticks = np.arange(0, p.shape[0], 100)
        ax0.set_xticks(x_ticks)
        ax0.set_xticklabels(x_ticks, rotation=90)
        ax0.xaxis.tick_top()
        ax0.set_yticks(y_ticks)
        ax0.set_yticklabels(y_ticks)
        ax0.tick_params(axis='both', which='both', labelsize=8)

        q = queries['epoch'+str(e)]['batch'+str(b)]
        ax1 = sns.heatmap(q, cmap='jet', ax=axes[1])
        axes[1].set_title('Query', fontsize=16)
        x_ticks = np.arange(0, p.shape[1], 100)
        y_ticks = np.arange(0, p.shape[0], 100)
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_ticks, rotation=90)
        ax1.xaxis.tick_top()
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels(y_ticks)
        ax1.tick_params(axis='both', which='both', labelsize=8)

        plt.suptitle("Projection Weights, Epoch "+str(e+1)+" Batch "+str(b+1), fontsize=20, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = "keys_VS_queries/e"+str(e+1)+'b'+str(b+1)+".jpg"
        plt.savefig(save_path, format="jpeg", dpi=300)

        # Remove the existing colorbar to reset it for the next iteration
        if (e>0) and (b>0):
            ax1.collections[0].colorbar.remove()
            ax0.collections[0].colorbar.remove()
        plt.clf()  # Clear the current figure to prepare for the next iteration

#-------- layer1 VS layer3 (key) ---------#

print("layer1 VS layer3")
for e in range(epochs):
    for b in tqdm(range(batches)):

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        p = keys1['epoch'+str(e)]['batch'+str(b)]
        ax0 = sns.heatmap(p, cmap='jet', ax=axes[0])
        axes[0].set_title('1st Layer', fontsize=22)
        x_ticks = np.arange(0, p.shape[1], 100)
        y_ticks = np.arange(0, p.shape[0], 100)
        ax0.set_xticks(x_ticks)
        ax0.set_xticklabels(x_ticks, rotation=90)
        ax0.xaxis.tick_top()
        ax0.set_yticks(y_ticks)
        ax0.set_yticklabels(y_ticks)
        ax0.tick_params(axis='both', which='both', labelsize=8)

        q = keys['epoch'+str(e)]['batch'+str(b)]
        ax1 = sns.heatmap(q, cmap='jet', ax=axes[1])
        axes[1].set_title('3rd Layer', fontsize=22)
        x_ticks = np.arange(0, p.shape[1], 100)
        y_ticks = np.arange(0, p.shape[0], 100)
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_ticks, rotation=90)
        ax1.xaxis.tick_top()
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels(y_ticks)
        ax1.tick_params(axis='both', which='both', labelsize=8)

        plt.suptitle("Key Weights, Epoch "+str(e+1)+" Batch "+str(b+1), fontsize=26, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = "layer_VS_layer/e"+str(e+1)+'b'+str(b+1)+".jpg"
        plt.savefig(save_path, format="jpeg", dpi=300)

        # Remove the existing colorbar to reset it for the next iteration
        if (e>0) and (b>0):
            ax1.collections[0].colorbar.remove()
            ax0.collections[0].colorbar.remove()
        plt.clf()  # Clear the current figure to prepare for the next iteration

#-------- layer1 VS layer3 Deltas (query) ---------#
print("layer1 VS layer3 deltas")

# 1
for b in tqdm(range(batches-1)):

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    o = queries1['epoch' + str(0)]['batch' + str(b)]
    n = queries1['epoch' + str(0)]['batch' + str(b + 1)]
    p = pd.DataFrame(n - o)
    ax0 = sns.heatmap(p, cmap='jet', ax=axes[0])
    axes[0].set_title('1st Layer', fontsize=22)
    x_ticks = np.arange(0, p.shape[1], 100)
    y_ticks = np.arange(0, p.shape[0], 100)
    ax0.set_xticks(x_ticks)
    ax0.set_xticklabels(x_ticks, rotation=90)
    ax0.xaxis.tick_top()
    ax0.set_yticks(y_ticks)
    ax0.set_yticklabels(y_ticks)
    ax0.tick_params(axis='both', which='both', labelsize=8)

    o = queries['epoch' + str(0)]['batch' + str(b)]
    n = queries['epoch' + str(0)]['batch' + str(b + 1)]
    q = pd.DataFrame(n - o)
    ax1 = sns.heatmap(q, cmap='jet', ax=axes[1])
    axes[1].set_title('3rd Layer', fontsize=22)
    x_ticks = np.arange(0, p.shape[1], 100)
    y_ticks = np.arange(0, p.shape[0], 100)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_ticks, rotation=90)
    ax1.xaxis.tick_top()
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_ticks)
    ax1.tick_params(axis='both', which='both', labelsize=8)

    plt.suptitle("Query Weight Deltas, Epoch "+str(1)+" Batch "+str(b+1), fontsize=26, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = "layer_VS_layer_delta/e"+str(1)+'b'+str(b+1)+".jpg"
    plt.savefig(save_path, format="jpeg", dpi=300)

    # Remove the existing colorbar to reset it for the next iteration
    if (e>0) and (b>0):
        ax1.collections[0].colorbar.remove()
        ax0.collections[0].colorbar.remove()
    plt.clf()  # Clear the current figure to prepare for the next iteration

# 2
for b in tqdm(range(batches-1)):

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    o = queries1['epoch' + str(1)]['batch' + str(b)]
    n = queries1['epoch' + str(1)]['batch' + str(b + 1)]
    p = pd.DataFrame(n - o)
    ax0 = sns.heatmap(p, cmap='jet', ax=axes[0])
    axes[0].set_title('1st Layer', fontsize=22)
    x_ticks = np.arange(0, p.shape[1], 100)
    y_ticks = np.arange(0, p.shape[0], 100)
    ax0.set_xticks(x_ticks)
    ax0.set_xticklabels(x_ticks, rotation=90)
    ax0.xaxis.tick_top()
    ax0.set_yticks(y_ticks)
    ax0.set_yticklabels(y_ticks)
    ax0.tick_params(axis='both', which='both', labelsize=8)

    o = queries['epoch' + str(1)]['batch' + str(b)]
    n = queries['epoch' + str(1)]['batch' + str(b + 1)]
    q = pd.DataFrame(n - o)
    ax1 = sns.heatmap(q, cmap='jet', ax=axes[1])
    axes[1].set_title('3rd Layer', fontsize=22)
    x_ticks = np.arange(0, p.shape[1], 100)
    y_ticks = np.arange(0, p.shape[0], 100)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_ticks, rotation=90)
    ax1.xaxis.tick_top()
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_ticks)
    ax1.tick_params(axis='both', which='both', labelsize=8)

    plt.suptitle("Query Weight Deltas, Epoch "+str(2)+" Batch "+str(b+1), fontsize=26, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = "layer_VS_layer_delta/e"+str(2)+'b'+str(b+1)+".jpg"
    plt.savefig(save_path, format="jpeg", dpi=300)

    # Remove the existing colorbar to reset it for the next iteration
    if (e>0) and (b>0):
        ax1.collections[0].colorbar.remove()
        ax0.collections[0].colorbar.remove()
    plt.clf()  # Clear the current figure to prepare for the next iteration

# 3
for b in tqdm(range(batches-1)):

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    o = queries1['epoch' + str(2)]['batch' + str(b)]
    n = queries1['epoch' + str(2)]['batch' + str(b + 1)]
    p = pd.DataFrame(n - o)
    ax0 = sns.heatmap(p, cmap='jet', ax=axes[0])
    axes[0].set_title('1st Layer', fontsize=22)
    x_ticks = np.arange(0, p.shape[1], 100)
    y_ticks = np.arange(0, p.shape[0], 100)
    ax0.set_xticks(x_ticks)
    ax0.set_xticklabels(x_ticks, rotation=90)
    ax0.xaxis.tick_top()
    ax0.set_yticks(y_ticks)
    ax0.set_yticklabels(y_ticks)
    ax0.tick_params(axis='both', which='both', labelsize=8)

    o = queries['epoch' + str(2)]['batch' + str(b)]
    n = queries['epoch' + str(2)]['batch' + str(b + 1)]
    q = pd.DataFrame(n - o)
    ax1 = sns.heatmap(q, cmap='jet', ax=axes[1])
    axes[1].set_title('3rd Layer', fontsize=22)
    x_ticks = np.arange(0, p.shape[1], 100)
    y_ticks = np.arange(0, p.shape[0], 100)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_ticks, rotation=90)
    ax1.xaxis.tick_top()
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_ticks)
    ax1.tick_params(axis='both', which='both', labelsize=8)

    plt.suptitle("Query Weight Deltas, Epoch "+str(3)+" Batch "+str(b+1), fontsize=26, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = "layer_VS_layer_delta/e"+str(3)+'b'+str(b+1)+".jpg"
    plt.savefig(save_path, format="jpeg", dpi=300)

    # Remove the existing colorbar to reset it for the next iteration
    if (e>0) and (b>0):
        ax1.collections[0].colorbar.remove()
        ax0.collections[0].colorbar.remove()
    plt.clf()  # Clear the current figure to prepare for the next iteration

#-------- query VS key Deltas (l1) ---------#
print("query VS key Deltas")

# 1
for b in tqdm(range(batches-1)):

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    o = queries1['epoch' + str(0)]['batch' + str(b)]
    n = queries1['epoch' + str(0)]['batch' + str(b + 1)]
    p = pd.DataFrame(n - o)
    ax0 = sns.heatmap(p, cmap='jet', ax=axes[0])
    axes[0].set_title('Query', fontsize=22)
    x_ticks = np.arange(0, p.shape[1], 100)
    y_ticks = np.arange(0, p.shape[0], 100)
    ax0.set_xticks(x_ticks)
    ax0.set_xticklabels(x_ticks, rotation=90)
    ax0.xaxis.tick_top()
    ax0.set_yticks(y_ticks)
    ax0.set_yticklabels(y_ticks)
    ax0.tick_params(axis='both', which='both', labelsize=8)

    o = keys1['epoch' + str(0)]['batch' + str(b)]
    n = keys1['epoch' + str(0)]['batch' + str(b + 1)]
    q = pd.DataFrame(n - o)
    ax1 = sns.heatmap(q, cmap='jet', ax=axes[1])
    axes[1].set_title('Key', fontsize=22)
    x_ticks = np.arange(0, p.shape[1], 100)
    y_ticks = np.arange(0, p.shape[0], 100)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_ticks, rotation=90)
    ax1.xaxis.tick_top()
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_ticks)
    ax1.tick_params(axis='both', which='both', labelsize=8)

    plt.suptitle("Query VS Key Deltas, Epoch "+str(1)+" Batch "+str(b+1), fontsize=26, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = "keys_VS_query_delta/e"+str(1)+'b'+str(b+1)+".jpg"
    plt.savefig(save_path, format="jpeg", dpi=300)

    # Remove the existing colorbar to reset it for the next iteration
    if (e>0) and (b>0):
        ax1.collections[0].colorbar.remove()
        ax0.collections[0].colorbar.remove()
    plt.clf()  # Clear the current figure to prepare for the next iteration

# 2
for b in tqdm(range(batches-1)):

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    o = queries1['epoch' + str(1)]['batch' + str(b)]
    n = queries1['epoch' + str(1)]['batch' + str(b + 1)]
    p = pd.DataFrame(n - o)
    ax0 = sns.heatmap(p, cmap='jet', ax=axes[0])
    axes[0].set_title('Query', fontsize=22)
    x_ticks = np.arange(0, p.shape[1], 100)
    y_ticks = np.arange(0, p.shape[0], 100)
    ax0.set_xticks(x_ticks)
    ax0.set_xticklabels(x_ticks, rotation=90)
    ax0.xaxis.tick_top()
    ax0.set_yticks(y_ticks)
    ax0.set_yticklabels(y_ticks)
    ax0.tick_params(axis='both', which='both', labelsize=8)

    o = keys1['epoch' + str(1)]['batch' + str(b)]
    n = keys1['epoch' + str(1)]['batch' + str(b + 1)]
    q = pd.DataFrame(n - o)
    ax1 = sns.heatmap(q, cmap='jet', ax=axes[1])
    axes[1].set_title('Key', fontsize=22)
    x_ticks = np.arange(0, p.shape[1], 100)
    y_ticks = np.arange(0, p.shape[0], 100)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_ticks, rotation=90)
    ax1.xaxis.tick_top()
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_ticks)
    ax1.tick_params(axis='both', which='both', labelsize=8)

    plt.suptitle("Query VS Key Deltas, Epoch "+str(2)+" Batch "+str(b+1), fontsize=26, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = "keys_VS_query_delta/e"+str(2)+'b'+str(b+1)+".jpg"
    plt.savefig(save_path, format="jpeg", dpi=300)

    # Remove the existing colorbar to reset it for the next iteration
    if (e>0) and (b>0):
        ax1.collections[0].colorbar.remove()
        ax0.collections[0].colorbar.remove()
    plt.clf()  # Clear the current figure to prepare for the next iteration

# 3
for b in tqdm(range(batches-1)):

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    o = queries1['epoch' + str(2)]['batch' + str(b)]
    n = queries1['epoch' + str(2)]['batch' + str(b + 1)]
    p = pd.DataFrame(n - o)
    ax0 = sns.heatmap(p, cmap='jet', ax=axes[0])
    axes[0].set_title('Query', fontsize=22)
    x_ticks = np.arange(0, p.shape[1], 100)
    y_ticks = np.arange(0, p.shape[0], 100)
    ax0.set_xticks(x_ticks)
    ax0.set_xticklabels(x_ticks, rotation=90)
    ax0.xaxis.tick_top()
    ax0.set_yticks(y_ticks)
    ax0.set_yticklabels(y_ticks)
    ax0.tick_params(axis='both', which='both', labelsize=8)

    o = keys1['epoch' + str(2)]['batch' + str(b)]
    n = keys1['epoch' + str(2)]['batch' + str(b + 1)]
    q = pd.DataFrame(n - o)
    ax1 = sns.heatmap(q, cmap='jet', ax=axes[1])
    axes[1].set_title('Key', fontsize=22)
    x_ticks = np.arange(0, p.shape[1], 100)
    y_ticks = np.arange(0, p.shape[0], 100)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_ticks, rotation=90)
    ax1.xaxis.tick_top()
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_ticks)
    ax1.tick_params(axis='both', which='both', labelsize=8)

    plt.suptitle("Query VS Key Deltas, Epoch "+str(3)+" Batch "+str(b+1), fontsize=26, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = "keys_VS_query_delta/e"+str(3)+'b'+str(b+1)+".jpg"
    plt.savefig(save_path, format="jpeg", dpi=300)

    # Remove the existing colorbar to reset it for the next iteration
    if (e>0) and (b>0):
        ax1.collections[0].colorbar.remove()
        ax0.collections[0].colorbar.remove()
    plt.clf()  # Clear the current figure to prepare for the next iteration

a=4