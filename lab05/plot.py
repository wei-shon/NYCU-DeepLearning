import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   # 資料視覺化套件
with open('./logs/fp/rnn_size=256-predictor-posterior-rnn_layers=2-1-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000/train_record.txt', newline='') as file:
    psnr = []
    kl_loss = []
    for line in file.readlines():
        if line[0]=='=':
            print("psnr : " , str(line[39:47]))
            psnr.append(float(line[39:47]))
        elif line[0]=='[':
            print("kl_loss : " , str(line[-9:-2]))
            kl_loss.append(float(line[-9:-2]))

    # plot PSNR
    epcoh = [i for i in range(61)]
    plt.rcParams['font.size'] = 14
    plt.figure(figsize=(7, 5))
    plt.title('Avarage PSNR Curve')
    plt.xlabel('epochs')
    plt.ylabel('PSNR')
    plt.plot(epcoh, psnr, color='blue', linestyle="-")
    plt.tight_layout()
    plt.savefig("./Diagram/PSNR.jpg") 
    plt.close()


    # plot PSNR
    epcoh = [i for i in range(300)]
    plt.rcParams['font.size'] = 14
    plt.figure(figsize=(7, 5))
    plt.title('KL Loss Curve')
    plt.xlabel('epochs')
    plt.ylabel('KL Loss')
    plt.plot(epcoh, kl_loss, color='blue', linestyle="-")
    plt.tight_layout()
    plt.savefig("./Diagram/KL_Loss.jpg") 
    plt.close()
