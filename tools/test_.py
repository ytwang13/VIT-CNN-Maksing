import torch
mask_ratio = 0.6
x = torch.rand([20,196,3])
N, L, D = x.shape  # batch, length, dim
out_x, out_y = 16, 16
mask_ratio = 0.6
block_size_x = int(out_x * mask_ratio)
block_size_y = int(out_y * mask_ratio)

start_x = torch.randint(0, out_x - block_size_x - 1, (N,))
start_y = torch.randint(0, out_y - block_size_y - 1, (N,))

# Generate the binary mask: 0 is keep, 1 is remove
mask = torch.zeros([N, out_x, out_y], device=x.device)

# ids_keep = start_x + start_y * out_x
for i in range(N):
    mask[i,start_x[i]:(start_x[i]+block_size_x),start_y[i]:(start_y[i]+block_size_y)]=1

mask = mask.flatten(1)

# Calculate ids_keep
ids_keep1 = torch.nonzero(mask, as_tuple=True)
ids_keep = torch.nonzero(mask, as_tuple=True)[1].view(N, -1)

# Calculate ids_restore
ids_restore = torch.argsort(ids_keep, dim=1)

# Calculate x_masked
expanded_ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
# x_masked = torch.gather(x, dim=1, index=expanded_ids_keep)



noise = torch.rand(5, 6, device='cuda')  # noise in [0, 1]

# # sort noise for each sample
ids_shuffle = torch.argsort(
    noise, dim=1)  # ascend: small is keep, large is remove
ids_restore = torch.argsort(ids_shuffle, dim=1)
print(111)