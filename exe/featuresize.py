import math

def compute_output_dim(input_dim, kernel_size, stride, padding):
    # input_dim: 输入尺寸（单一方向，如宽或高）
    # kernel_size: 卷积或池化核大小
    # stride: 步幅
    # padding: 填充
    # 返回输出尺寸（单一方向）
    return math.floor((input_dim + 2*padding - kernel_size) / stride) + 1

layers = [
    {'type': 'conv', 'kernel': 3, 'stride': 1, 'padding': 0, 'channel': 16},
    {'type': 'pool', 'kernel': 2, 'stride': 2, 'padding': 0},
    # {'type': 'conv', 'kernel': 5, 'stride': 1, 'padding': 0, 'channel': 400},
    # {'type': 'pool', 'kernel': 1, 'stride': 1, 'padding': 1, 'channel': 400},
]

# 初始输入尺寸: [H, W, C]
image_size = [512, 512, 3]

H, W, C = image_size

for i, layer in enumerate(layers, 1):
    if layer['type'] == 'conv':
        # 计算输出高度和宽度
        H_out = compute_output_dim(H, layer['kernel'], layer['stride'], layer['padding'])
        W_out = compute_output_dim(W, layer['kernel'], layer['stride'], layer['padding'])
        C_out = layer['channel']  # 通道数由该层的定义决定
        H, W, C = H_out, W_out, C_out

    elif layer['type'] == 'pool':
        # 池化层与卷积相同公式
        H_out = compute_output_dim(H, layer['kernel'], layer['stride'], layer['padding'])
        W_out = compute_output_dim(W, layer['kernel'], layer['stride'], layer['padding'])
        # 池化层不改变通道数
        H, W, C = H_out, W_out, C

    print(f"After layer {i} ({layer['type']}): {H}x{W}x{C}")

# 最终输出
print("Final output size:", [H, W, C])
