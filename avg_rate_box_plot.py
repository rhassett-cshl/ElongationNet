import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

nucleotide_to_index = {
    'A': 0,
    'T': 1,
    'G': 2,
    'C': 3
}

df = pd.read_csv('elongation_net_results_final.csv')

nucleotides = ['A', 'T', 'G', 'C']
num_seq_features = len(nucleotides)

# kmer mapped to target pos of active site  
kmer_dict = {"CA": 0, "GCA": 1, "AGT": 1, "CT": 0, "GCT": 1, "TGC": 1, "TAC": 1, "TTG": 1, "AA": 0, "TAG": 1, "TGG": 1, "TAA": 1, "TTT": 1}

for motif, active_site_idx in kmer_dict.items():
    motif_length = len(motif)
    
    # extract starting indices of motif in dataframe
    motif_indices = []
    for i in range(len(df) - motif_length + 1):  
        if all(df.loc[i + j, motif[j]] == 1 for j in range(motif_length)):
            motif_indices.append(i)
    print(len(motif_indices))
    
    # extract active site index of each motif
    motif_target_indices = [x + active_site_idx for x in motif_indices]
    
    # extract cnn and glm zetas at those positions
    cnn_elongation_rates = df.loc[motif_target_indices, 'Predicted_Zeta'].tolist()
    glm_elongation_rates = df.loc[motif_target_indices, 'GLM_Combined_Zeta'].tolist()

    # plot box plot for those rates
    df2 = pd.DataFrame({
        'Values': cnn_elongation_rates + glm_elongation_rates,
        'Group': ['CNN'] * len(cnn_elongation_rates) + ['GLM'] * len(glm_elongation_rates)
    })

    sns.boxplot(x='Group', y='Values', data=df2, showfliers=False)
    plt.title(f"Box Plot for {motif} Elongation Rates")
    plt.show()
    plt.savefig(f"box_plot_results/{motif}_boxplot.png")
    plt.clf()