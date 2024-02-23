import matplotlib.pyplot as plt

def plot_loss_curves(neural_net_train_loss, neural_net_valid_loss, glm_valid_loss, config_filename):
    epochs = range(1, len(loss_neural_net_train) + 1)
    plt.plot(epochs, loss_neural_net_train, label='train_neural_net_loss')
    plt.plot(epochs, loss_neural_net_valid, label='valid_neural_net_loss')
    plt.plot(epochs, loss_glm_valid, label='valid_glm_loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(results_folder_path + config_file_name + "/loss_curves.png")
    plt.show 
