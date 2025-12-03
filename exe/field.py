def compute_input_receptive_field(layers, target_point):
    """
    计算卷积神经网络中输出点对应到输入图像的像素范围。

    参数:
        layers (list of dict): 每层的参数列表，支持 'type', 'kernel', 'stride', 'padding'。
            - 'type': 层类型 ('conv' 或 'pool')
            - 'kernel': 卷积核或池化核大小 (int 或 tuple)
            - 'stride': 步幅 (int)
            - 'padding': 填充大小 (int)
        target_point (tuple): 输出的目标像素坐标 (row, col)
    返回:
        (row_range, col_range): 输入图像中的行和列范围
    """
    # 初始化目标点的感受野范围
    row, col = target_point
    row_start, row_end = row, row
    col_start, col_end = col, col

    # 反向遍历网络层
    for layer in reversed(layers):
        # 处理卷积层
        if layer['type'] == 'conv':
            # 处理kernel
            kernel = layer['kernel'] if isinstance(layer['kernel'], tuple) else (layer['kernel'], layer['kernel'])
            kernel_h, kernel_w = kernel
            stride = layer['stride']
            padding = layer['padding']

            # 扩展感受野范围
            row_start = row_start * stride - padding
            row_end = row_end * stride + kernel_h - 1 - padding
            col_start = col_start * stride - padding
            col_end = col_end * stride + kernel_w - 1 - padding

        # 处理池化层
        elif layer['type'] == 'pool':
            # 处理kernel
            kernel = layer['kernel'] if isinstance(layer['kernel'], tuple) else (layer['kernel'], layer['kernel'])
            kernel_h, kernel_w = kernel
            stride = layer['stride']
            padding = layer['padding']

            # 扩展感受野范围
            row_start = row_start * stride - padding
            row_end = row_end * stride + kernel_h - 1 - padding
            col_start = col_start * stride - padding
            col_end = col_end * stride + kernel_w - 1 - padding

    return ((row_start, row_end), (col_start, col_end))

def map_input_point_to_feature_map(layers, input_point):
    """
    将原始输入图像中的像素点映射到最终输出特征图中的位置。

    参数:
        layers (list of dict): 网络层参数列表，支持 'type', 'kernel', 'stride', 'padding'。
            - 'type': 层类型 ('conv' 或 'pool')
            - 'kernel': 卷积核或池化核大小 (int 或 tuple)
            - 'stride': 步幅 (int)
            - 'padding': 填充大小 (int)
        input_point (tuple): 输入图像中的原始像素坐标 (row, col)

    返回:
        (row, col): 输出特征图中对应的像素坐标
    """
    # 初始化输入点
    row, col = input_point

    # 正向遍历网络层
    for layer in layers:
        # 处理卷积层
        if layer['type'] == 'conv':
            # 处理kernel
            kernel = layer['kernel'] if isinstance(layer['kernel'], tuple) else (layer['kernel'], layer['kernel'])
            kernel_h, kernel_w = kernel
            stride = layer['stride']
            padding = layer['padding']

            # 考虑padding的影响
            row += padding
            col += padding

            # 计算在该层的输出位置
            # 对于卷积层，使用整除计算输出坐标
            row = row // stride
            col = col // stride

        # 处理池化层
        elif layer['type'] == 'pool':
            # 处理kernel
            kernel = layer['kernel'] if isinstance(layer['kernel'], tuple) else (layer['kernel'], layer['kernel'])
            kernel_h, kernel_w = kernel
            stride = layer['stride']
            padding = layer['padding']

            # 考虑padding的影响
            row += padding
            col += padding

            # 计算在该层的输出位置
            # 对于池化层，使用整除计算输出坐标
            row = row // stride
            col = col // stride

    return (row, col)

def main():
    # 定义网络层配置（与之前的例子相同）
    layers = [
        {'type': 'conv', 'kernel': 3, 'stride': 1, 'padding': 1},
        {'type': 'conv', 'kernel': 3, 'stride': 1, 'padding': 1},
        {'type': 'pool', 'kernel': 2, 'stride': 2, 'padding': 0},
    ]

    # 测试输入图像中的一个点
    input_point = (4, 6)

    # 计算这个点在输出特征图中的位置
    output_point = map_input_point_to_feature_map(layers, input_point)
    print(f"输入图像中的点 {input_point} 映射到输出特征图的位置: {output_point}")

    # 反向验证
    import sys
    sys.path.append('.')  # 确保可以导入之前定义的函数

    # 检查映射点是否在计算出的感受野范围内
    receptive_field = compute_input_receptive_field(layers, output_point)
    row_range, col_range = receptive_field

    print("\n感受野验证:")
    print(f"感受野行范围: {row_range}")
    print(f"感受野列范围: {col_range}")

    # 检查输入点是否在感受野范围内
    row_in_range = row_range[0] <= input_point[0] <= row_range[1]
    col_in_range = col_range[0] <= input_point[1] <= col_range[1]

    print(f"\n输入点 {input_point} 是否在感受野范围内:")
    print(f"行: {row_in_range}")
    print(f"列: {col_in_range}")


    # # 目标输出点坐标
    # target_point = (1, 2)
    # # 计算输入像素范围
    # input_range = compute_input_receptive_field(layers, target_point)
    # print("输入行范围:", input_range[0])
    # print("输入列范围:", input_range[1])

if __name__ == '__main__':
    main()
