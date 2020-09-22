# from utils import create_input_files
#
# if __name__ == '__main__':
#     # Create input files (along with word map)
#     create_input_files(dataset='flickr8k',
#                        karpathy_json_path=r"D:\Users\wt\Downloads\caption_datasets\dataset_flickr8k.json",
#                        image_folder=r"D:\Users\wt\Downloads\image_caption\flickr8k\images",
#                        captions_per_image=5,
#                        min_word_freq=5,
#                        output_folder=r"D:\Users\wt\Downloads\image_caption\outputfile",
#                        max_len=50)
#
# import torch
# import torch.nn as nn
# import numpy as np
# from torch.optim import SGD
#
# X = np.array([[0, 0],
#               [0, 1],
#               [1, 0],
#               [1, 1]])
# y = np.array([[1, 0, 1, 0]]).reshape(4,1)
#
# X = torch.tensor(X).float()
# y = torch.tensor(y).float()
#
# Net = torch.nn.Sequential(
#     nn.Linear(2, 10),
#     nn.ReLU(),
#     nn.Linear(10, 1),
#     nn.Sigmoid()
# )
#
# optimizer = SGD(Net.parameters(), lr=0.05)
# loss_func = nn.MSELoss()
# for epoch in range(5000):
#     out = Net(X)
#     loss = loss_func(out, y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#
#
#


