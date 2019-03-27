import matplotlib.pyplot as plt
import os

def extraData(file_name):
    data_file = open(file_name, 'r')
    dataset = data_file.readlines()
    x = []
    y = []
    for data in dataset:
        data = data.strip().split(',')
        x.append(float(data[0]))
        y.append(float(data[1]))
    data_file.close()
    return x,y



if __name__ == "__main__":
    pca_root = "./results/pca/"
    raw_root = "./results/raw/"
    
    figsize = 10, 12
    plt.subplots(figsize=figsize)
    ax1 = plt.subplot(2, 1, 1)
    plt.ylim(0.0, 1.0)
    ax1.set_title("Without PCA")
    raw_file_names = os.listdir(raw_root)
    for raw_file_name in raw_file_names:
        eigen_len = raw_file_name[:-4]
        raw_file = raw_root + raw_file_name
        x,y = extraData(raw_file)
        plt.plot(x, y, label="Dimension: "+eigen_len)
    plt.legend(loc='lower right')
    plt.xlabel("k value")
    plt.ylabel("average accuracy")
    ax2 = plt.subplot(2, 1, 2)
    plt.ylim(0.0, 1.0)
    ax2.set_title("With PCA")
    pca_file_names = os.listdir(pca_root)
    for pca_file_name in pca_file_names:
        eigen_len = pca_file_name[:-4]
        pca_file = pca_root + pca_file_name
        x,y = extraData(pca_file)
        plt.plot(x, y, label="Dimension: "+eigen_len)
    plt.legend(loc='lower right')
    plt.xlabel("k value")
    plt.ylabel("average accuracy")
    plt.show()
