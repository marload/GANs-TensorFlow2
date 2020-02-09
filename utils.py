import matplotlib.pyplot as plt
from datetime import datetime


def generate_result_images(model, step, z_input):
    predictions = model(z_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.axis('off')

    plt.savefig('./result/{}_{}.png'.format(datetime.now(), step))
