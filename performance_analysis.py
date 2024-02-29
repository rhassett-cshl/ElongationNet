import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def plot_data(nn_zeta, glm_zeta, filename, gene_id):
    indices = range(len(nn_zeta))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(indices, nn_zeta, color='blue', label='Neural Net Zeta', alpha=0.5)
    ax.plot(indices, glm_zeta, color='orange', label='GLM Zeta', alpha=0.5)
    
    ax.set_title(f'{gene_id} Neural Net vs GLM Combined Model Zeta')
    ax.set_xlabel('Index')
    ax.set_ylabel('Zeta')
    ax.legend()
    
    plt.ylim(-5, 40)

    plt.savefig(filename)
    

config_name = 'cnn_performance_analysis'

metrics_file = f'./results/{config_name}/{config_name}_metrics.txt'

x_predictions_file = f'./results/{config_name}/{config_name}_xji_predictions.csv'

zeta_plot1 = f'./results/{config_name}/{config_name}_plot1.png'

zeta_plot2 = f'./results/{config_name}/{config_name}_plot1.png'

df = pd.read_csv(f"./results/{config_name}/{config_name}_results.csv")

print(df.head())

genes = df.groupby('GeneId').groups.keys()
print(genes)

df_by_gene = list(df.groupby('GeneId'))

plot_data(df_by_gene[0][1]["Predicted_Zeta"], df_by_gene[0][1]["GLM_Combined_Zeta"], zeta_plot1, df_by_gene[0][0])

plot_data(df_by_gene[1][1]["Predicted_Zeta"], df_by_gene[1][1]["GLM_Combined_Zeta"], zeta_plot2, df_by_gene[1][0])

metrics = []

cnn_avg_loss = f"Average Loss: {df['CNN_Loss'].mean()}\n"
cnn_sum_loss = f"Summed Loss: {df['CNN_Loss'].sum()}\n"
glm_avg_loss = f"Average Loss: {df['GLM_Loss'].mean()}\n"
glm_sum_loss = f"Summed Loss: {df['GLM_Loss'].sum()}\n"

with open(metrics_file, 'a') as f:
    f.write(cnn_avg_loss)
    f.write(cnn_sum_loss)
    f.write(glm_avg_loss)
    f.write(glm_sum_loss)


df['C_j/Zeta'] = df['C_j'] / df['Predicted_Zeta']
aggregated_data = df.groupby('GeneId').agg(
    Sum_X_ji=pd.NamedAgg(column='X_ji', aggfunc='sum'),
    Sum_C_j_Zeta=pd.NamedAgg(column='C_j/Zeta', aggfunc='sum')
).reset_index()
aggregated_data.to_csv(x_predictions_file, index=False)


mse = ((df['X_ji'] - df['C_j/Zeta']) ** 2).mean()
r2 = r2_score(df['X_ji'], df['C_j/Zeta'])
zeta_description = df["Predicted_Zeta"].describe()

with open(metrics_file, 'a') as f:
    f.write(f"MSE {mse}\n")
    f.write(f"R2 {r2}\n")
    f.write(f"MSE/R2 {mse / r2}\n")
    f.write(zeta_description.to_string())