import torch, sys

input = sys.argv[1]
in_file_path = "/home/eunbi/softmax/data/" + input + "_data.txt"

inputs = torch.load(in_file_path)
torch.set_printoptions(profile="full", precision=6, sci_mode=False)
print(inputs)
