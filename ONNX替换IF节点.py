import onnx

# 加载ONNX模型
model_path = 'tmp/maskrcnn.onnx'  # 替换为你的ONNX模型路径
model = onnx.load(model_path)

# 遍历模型中的所有节点
for i, node in enumerate(model.graph.node):
    
    if node.op_type!="If":
        continue
    if_node_to_replace = node
    # then_branch_graph = node.attribute[0].g
    # 替换If节点，将其输入连接到所选分支的输入
    # 这里我们假设If节点的输入和所选分支的输入是匹配的
    # 你可能需要根据你的模型进行调整
    then_branch_graph = if_node_to_replace.attribute[0].g
    print(if_node_to_replace.attribute)
    # # 替换If节点，将其输入连接到所选分支的输入
    # # 这里我们假设If节点的输入和所选分支的输入是匹配的
    # # 你可能需要根据你的模型进行调整
    # for input_edge in if_node_to_replace.input:
    #     for node in then_branch_graph.node:
    #         for index, input_name in enumerate(node.input):
    #             if input_name == if_node_to_replace.input[0]:
    #                 node.input[index] = input_edge

    # # 将所选分支的输出连接到原If节点的输出
    # # 这里我们假设If节点的输出和所选分支的输出是匹配的
    # # 你可能需要根据你的模型进行调整
    # for output_edge in if_node_to_replace.output:
    #     for node in then_branch_graph.node:
    #         for index, output_name in enumerate(node.output):
    #             if output_name == if_node_to_replace.output[0]:
    #                 node.output[index] = output_edge
    
    #  # 将所选分支的节点添加到主图中
    # for node in then_branch_graph.node:
    #     model.graph.node.append(node)

    #  # 移除原If节点
    # model.graph.node.remove(if_node_to_replace)
    

# # 保存修改后的模型
# modified_model_path = 'tmp/modified_maskrcnn.onnx'
# onnx.save(model, modified_model_path)
# # print(f'Modified model saved to {modified_model_path}')