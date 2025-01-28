from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

column_names = ["motif", "cnn_avg_zeta", "glm_avg_zeta"]
avg_zeta_df = pd.DataFrame(columns=column_names)

nucleotide_to_index = {
    'A': 0,
    'T': 1,
    'G': 2,
    'C': 3
}

df = pd.read_csv('elongation_net_results_final.csv')

nucleotides = ['A', 'T', 'G', 'C']
num_seq_features = len(nucleotides)

motif_length = 4
active_site_idx = 1
kmer_type = f"{motif_length}-mers"
kmers = np.array([''.join(c) for c in product('ATGC', repeat=motif_length)])

for motif in kmers:
    # extract starting indices of motif in dataframe
    motif_indices = []
    for i in range(len(df) - motif_length + 1):  
        if all(df.loc[i + j, motif[j]] == 1 for j in range(motif_length)):
            motif_indices.append(i)
    print(len(motif_indices))
    
    # extract active site index of each motif
    motif_target_indices = [x + active_site_idx for x in motif_indices]
    
    # extract avg cnn and glm zetas at those positions
    kmer_avg_cnn_zeta = np.mean(df.loc[motif_target_indices, 'Predicted_Zeta'].tolist())
    kmer_avg_glm_zeta = np.mean(df.loc[motif_target_indices, 'GLM_Combined_Zeta'].tolist())

    # save results to csv file
    avg_zeta_df = avg_zeta_df.append({"motif": motif, "cnn_avg_zeta": kmer_avg_cnn_zeta, "glm_avg_zeta": kmer_avg_glm_zeta}, ignore_index=True)

avg_zeta_df.to_csv(f"motifs_zeta_{kmer_type}.csv", index=False)

x = avg_zeta_df['glm_avg_zeta']
y = avg_zeta_df['cnn_avg_zeta']

plt.plot([0, 10], [0, 10], 'r--')

# Calculate R^2 by squaring the Pearson correlation coefficient
r = np.corrcoef(x, y)[0, 1]
r2 = r ** 2

# Display R^2 on the plot
plt.text(0.05, 0.9, f"$R^2$ = {r2:.2f}", transform=plt.gca().transAxes,
         fontsize=22, color="blue")


sns.scatterplot(data=avg_zeta_df, x='glm_avg_zeta', y='cnn_avg_zeta')
plt.xlabel('GLM Average Zeta', fontsize=15)
plt.ylabel('CNN Average Zeta', fontsize=15)
plt.title(f'{kmer_type}', fontsize=24)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(f"Updated_Avg_Elongation_Rate_CNN_vs_GLM_{kmer_type}", dpi=300)