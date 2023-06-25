from matplotlib import pyplot as plt
import numpy as np
import os

def plot_axis(X, Y, title, color):
    plt.plot(X, Y, color=color)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True)
    # save plot to file
    plt.savefig("/home/octavio/dev/plot_prediction_from_subtitles/plots/" + title + ".png")
    plt.clf()

def plot_metrics():
    # read csv files from /home/octavio/dev/plot_prediction_from_subtitles/data/metrics_csv
    # and plot them
    labels = {
      "t5-rouge2.csv": "ROUGE-2",
      "t5-rougel.csv": "ROUGE-L",
      "t5-rouge1.csv": "ROUGE-1",
      "t5-rougelsum.csv": "ROUGE-Lsum",
    }
    
    for idx, file_name in enumerate(os.listdir("/home/octavio/dev/plot_prediction_from_subtitles/data/metrics_csv")):
        print(file_name)
        file_path = "/home/octavio/dev/plot_prediction_from_subtitles/data/metrics_csv/" + file_name
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1, names=['Step', 'Value'])
        plot_axis(data['Step'], data['Value'], labels[file_name], 'blue')

if __name__ == '__main__':
    plot_metrics()