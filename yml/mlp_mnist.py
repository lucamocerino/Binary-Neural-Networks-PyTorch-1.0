no_cuda: False
checkpoint: "results/mlp_mnist" 
filename: null
pretrained: null

model : "mlp"
save_path: "results/mlp_mnist" 
dataset : "mnist"
batch_size: 128 
test_batch_size: 100
optimizer: 'adam' 
lr: 0.01 
gamma: 0.1 
steps: [100, 200] 
epochs: 300 

