import torch
from torch import nn, optim # 引入神经网络和优化器(SGD随机梯度Adam自适应矩阵)


class LoRA(nn.Module):
    # A(1000*8)B(8*1000)，in_features是输入维度(也就是A的1000)，out_features是输出维度(也就是B的1000)
    # rank就是8
    def __init__(self, in_features, out_features, rank):
        super(LoRA,self).__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        # AB矩阵都要训练，所以初始化权重
        # A高斯初始化
        self.A.weight.data.normal_(mean=0,std=0.02)
        # B初始化为0的原因是，保证一开始的AB旁支跟没有是一样的
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x)) # 先乘A再乘B


# 将LoRA嫁接到原本的模型上
def apply_lora(model, rank=8):
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        # 如果这层是线性层，且是方阵
        if isinstance(module,nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0],module.weight.shape[1],rank).to(device)

            # 命名一下lora层，以后可通过module.lora访问
            setattr(module,'lora',lora)
            # 原始的输出
            original_forward = module.forward
            # 原始的输出+lora的输出
            def forward_with_lora(x,layer1=original_forward,layer2=lora):
                return layer1(x)+layer2(x)


def load_lora(model,path):
    # 保证lora与模型相同设备
    device = next(model.parameters()).device
    # 从文件中加载lora权重
    state_dict = torch.load(path, map_location=device)
    # 去掉module.这7个字符
    state_dict = {
        (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }

    # 遍历每一层
    for name, module in model.named_modules():
        # 如果是lora层
        if hasattr(module, "lora"):
            # 如果键名为 {name}.lora.lora_A 则变为lora_A
            lora_state = {
                k.replace(f"{name}.lora.", ""): v
                for k, v in state_dict.items()
                if f"{name}.lora." in k
            }
            # 把lora_state的权重放到了module.lora层上
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    # 处理包装模型，用来获取原始模型
    raw_model = getattr(model, "_orig_mod", model)
    # 用来存lora权重的字典
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            # 去掉前面module.这7个字符
            clean_name = name[7:] if name.startswith("module.") else name
            # 保存为 {name}.lora.lora_A: 权重A
            lora_state = {
                f"{clean_name}.lora.{k}": v for k, v in module.lora.state_dict().items()
            }
            # 都加入总字典
            state_dict.update(lora_state)
    torch.save(state_dict, path)

# 只有包装模型才会有module.这7个前缀
# 如多GPU训练(nn.DataParallel)
# 量化包装(torch.quantization.quantize_dynamic)
