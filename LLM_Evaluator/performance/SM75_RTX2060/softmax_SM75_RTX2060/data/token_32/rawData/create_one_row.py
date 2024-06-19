import torch, sys

input = sys.argv[1]
in_file_path = "/home/eunbi/softmax/data/token_32/rawData/" + input + "_data.txt"
out_file_path = "/home/eunbi/softmax/data/token_32/rawData/" + "softmax_" + input + ".txt"

input_tensor = torch.load(in_file_path)
torch.set_printoptions(profile="full", precision=6, sci_mode=False)

# Flatten the tensor into a single row
flat_data = input_tensor.view(-1).tolist()

# Write the flattened data to softmax_input.txt
with open(out_file_path, 'w') as f:
    f.write(' '.join(map(str, flat_data)))
