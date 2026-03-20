# 可以看到在pretrain的数据集就是开始结束开始结束，很粗暴的内容
# sft数据就是conversation内容，有严格标注，哪些是用户提问，哪些是需要会的回答
from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset
# 设置tokenizers不并行加速，避免报错
os.environ['TOKENIZERS_PARALLELISM'] = "false" # tokenizers_parallelism

# 进行数据增强
def pre_processing_chat(conversations, add_system_ratio=0.2):
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 第一条消息不空且role不是system
    if conversations and conversations[0].get('role') != 'system':
        # 有0.2的概率
        if random.random() < add_system_ratio:
            # 从system_prompts中随机选择一个角色设定放在最前面
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

# 数据后处理：清理模板渲染后多余的空<think>块
def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


# SFT训练的数据处理部分
class SFTDataset(Dataset):
    # 需要实现三个内定方法 1.load_data 2.__len__ 3.__getitem__
    # 序列输入最长为1024
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer=tokenizer # 分词器
        self.max_length=max_length
        # 加载训练集
        self.samples=self.load_dataset("json", data_files=jsonl_path, split="train")
        # 从AI助手assistant开始回答前打上开始标记，结束再打上结束标记
        # .input_ids文本转数字列表
        self.bos_id=tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id=tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    # 返回数据集的长度
    def __len__(self):
        return len(self.samples)


    # 对话转文本
    # [
    #     {"role": "user", "content": "北京天气怎么样？"},
    #     {"role": "assistant", "content": "今天北京晴，25度。"}
    # ]
    # user：北京天气怎么样？
    # assistant：今天北京晴，25度。
    def create_chat_prompt(self, conversations):
        # 拷贝一份原始数据
        messages = conversations.copy()
        # 看是否调用工具，如果有function call的话
        tools = (
            conversations[0]["function"]
            if(
                conversations
                and conversations[0].get("role") != "system"
                and conversations[0].get("function")
            ) else None
        )
        return self.tokenizer.apply_chat_template(messages, tokenizer=False, add_generation_prompt=False, tools=tools)


    # 遮羞布遮住instruction，只留AI回答的部分（为了计算AI的output和真实output的loss）
    # input_ids是AI回答的一段很长的数字序列
    def generate_labels(self, input_ids):
        # 让所有的label变为-100，-100是交叉熵损失函数默认忽视的值
        labels = [-100]*len(input_ids)
        # 找AI回答部分
        i = 0
        while i < len(input_ids):
            if input_ids[i:i+len(self.bos_id)] == self.bos_id: # 找到了开始标志，让start和end从下一个位置开始
                start = i+len(self.bos_id)
                end = start
                while end < len(input_ids): # 不超过总长前提下，end不断后移一位
                    if input_ids[end:end+len(self.eos_id)] == self.eos_id: # 找到了结束标志
                        break
                    end+=1

                # AI的回答+结束标记都恢复成原本的label
                for j in range(start, min(end+len(self.eos_id), len(input_ids))):
                    # 不用遮的地方
                    labels[j] = input_ids[j]
                # 跳到结束标志后再开始或结束
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels


    def __getitem__(self, index):
        sample = self.samples[index]
        # 是否随机插入system_prompt
        conversations = pre_processing_chat(sample['conversations'])
        # 1.把对话转换成文本
        prompt = self.create_chat_prompt(conversations)

        # 清空think块
        prompt = post_processing_chat(prompt)
        # tokenize截断，并且补足pad
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id]*(self.max_length-len(input_ids))

        # 2.遮羞布，生成标签
        labels = self.generate_labels(input_ids)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
