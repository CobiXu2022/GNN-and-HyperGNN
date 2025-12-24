import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

def create_animation(data_path, output_path):

    metrics_data = pd.read_csv(data_path)

    model_names = metrics_data['Model'].unique()
    epochs = metrics_data['Epoch'].unique()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(epochs[0], epochs[-1])
    ax.set_ylim(0.6, 1) 
    ax.set_title('Model Accuracy Over Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')

    colors = sns.color_palette()
    lines = {}
    for i, name in enumerate(model_names):
        line, = ax.plot([], [], color=colors[i], label=name) 
        lines[name] = line

    ax.legend(loc='lower right')

    def update(frame):
        for name in model_names:
            current_data = metrics_data[metrics_data['Model'] == name]
            lines[name].set_data(current_data['Epoch'][:frame + 1], current_data['Accuracy'][:frame + 1])
        return lines.values()


    ani = FuncAnimation(fig, update, frames=len(epochs), repeat=False)

    ani.save(output_path, writer='ffmpeg', fps=15, extra_args=['-vcodec', 'libx264'])
    print(f'Animation saved as {output_path}')

if __name__ == "__main__":
    data_path = "./data/acc_rec.csv"
    output_path = "./data/accuracy_animation.mp4"
    create_animation(data_path, output_path)