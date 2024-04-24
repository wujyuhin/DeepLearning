import torch
from datasets import load_dataset, load_from_disk
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import os
import argparse
from transformers import WEIGHTS_NAME, CONFIG_NAME

"""
整个大框架：
1，数据部分: raw data ===> encoding
    1.1 数据准备（构造Dataset类，继承Dataset类，包含__len__, 和__getitem()___ 方法）
    1.2 数据加载（构造Dataloader类，重要参数包括dataset，collator_fn, batch_size, shuffle, drop_last）
2，模型部分
    2.1 加载模型 （从预训练模型或者模型断点） 
    2.2 定义自己的模型类（例如下游任务，包括初始化和forward方法）
3，训练部分
    3.0 模型转入训练状态：model.train()
    3.1 选择优化器optimizer（重要参数包括：模型需要训练的参数params，model.parameters(), 学习率lr）
    3.2 选择调度器scheduler (重要参数包括：优化器optimizer，训练步数num_training_steps, num_warmup_steps)
    3.3 定义损失函数loss
    3.4 梯度回传 loss.backward()
    3.5 优化器更新 optimizer.step()
    3.6 调度器更新 scheduler.step()
    3.7 梯度置零 optimizer.zero_grad()
    3.8 保存checkpoint：
        定义一个名称为checkpoint的字典，里面包括要存储的量，例如模型的状态参数、优化器的状态参数、步数或epoch数、损失等。
        例如：checkpoint = {"model_state_dict": model.state_dict(),
                           "optimizer_state_dict": optimizer.state_dict()，
                           "step": i,
                           "epoch": epoch,
                           "loss": loss}
        再保存该checkpoint：torch.save(checkpoint, path)
    3.9 保存训练完后的模型：
        a, 保存整体模型结构和参数：torch.save(model, path)
        b, 仅保存模型参数: torch.save(model.state_dict(), path)
    3.10 载入模型 
        如果存储的是模型（和优化器）状态参数，则首先要初始化对应的模型和优化器，然后通过：model.load_state_dict(torch.load(path)) 来载入
        模型状态参数（optimizer类似）。
        如果存储的是模型的整体结构和状态参数，则不需要初始化，直接载入整个模型：model = torch.load(path)
4, 推理
   4.1 模型转为推理状态：model.eval()
"""

parser = argparse.ArgumentParser(description='config')
parser.add_argument("-ml", "--max_length", default=512, type=int, help='max_length')
parser.add_argument('-bs', "--batch_size", default=8, type=int, help='batch_size')
parser.add_argument('-out', "--output_dir", type=str, help="model directory")
parser.add_argument('-lr', "--learning_rate", default=5e-4, type=float, help="learning_rate")

args = parser.parse_args(args=[])

print(args.batch_size)
print(args.output_dir)

# step1 : prepare dataset

#dataset = load_dataset(path='sst2')
#dataset.save_to_disk(dataset_dict_path='./data/SST2')
# dataset 里面通常包括train，validation，test三个数据集，每一个都是一个Dataset类
dataset = load_from_disk(dataset_path='./data/SST2')
# 可通过key的方式，类似字典获得相应集合的数据
dataset['train'] = dataset['train'].select(range(1000))
train_dataset = dataset['train']
validation_dataset = dataset['validation']


# step2: preprocessing
def pre_process(data):
    data['sentence'] = ['[CLS] ' + s + '[SEP]' for s in data['sentence']]
    return data


# 这里dataset是DatasetDict 类，该类中也有map方法，该map方法对DatasetDict 中的key 和 value（Dataset类）进行遍历， 并
# 对每一个value 采用map方法（其实就是运行Dataset类中的map方法）
mapped_dataset = dataset.map(function=pre_process, batched=True, batch_size=100)

# step3: encoding by tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
embed = tokenizer.batch_encode_plus(batch_text_or_text_pairs=mapped_dataset['train']['sentence'][:2],
                                    add_special_tokens=False,  # add_special_tokens 会给输入加上[cls], [sep] 等特殊token
                                    padding="max_length",
                                    return_tensors='pt',
                                    max_length=512,  # 512 是默认值
                                    return_length=False)
for k, v in embed.items():
    print(k, v.shape)


# step4: 定义数据集加载器
def collator_fn(data):
    sents = [i['sentence'] for i in data]
    labels = [i['label'] for i in data]
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                       add_special_tokens=False,  # add_special_tokens 会给输入加上[cls], [sep] 等特殊token
                                       padding="max_length",
                                       return_tensors='pt',
                                       max_length=512,  # 512 是默认值
                                       return_length=False)
    input_ids = data['input_ids']
    token_type_ids = data['token_type_ids']
    attention_mask = data['attention_mask']
    return input_ids, token_type_ids, attention_mask, labels


# 注意：dataset要是Dataset类，其中包括了__getitem__这个内构函数，collator_fn 函数中的数据类型要和__getitem__方法取出
# 的数据结构一致
train_data_loader = DataLoader(dataset=train_dataset,
                               batch_size=8,
                               shuffle=True,
                               drop_last=False,  # 不够8个的是否丢弃
                               collate_fn=collator_fn)
len(train_data_loader)

# for i, (inputs, token_type, attention_mask, labels) in enumerate(train_data_loader):
#     print(i, inputs.shape)
#     print(i, token_type.shape)
#     print(i, attention_mask.shape)
#     print(i, labels)
#     break


# step5：导入模型和模型配置
pre_trained = BertModel.from_pretrained('bert-base-uncased')
for parameter in pre_trained.parameters():
    parameter.requires_grad_(False)


# out = pre_trained(input_ids = inputs,
#                   token_type_ids = token_type,
#                   attention_mask = attention_mask)
#out 中包括以下两部分
#out.last_hidden_state.shape     # [batch_size, seq_length, 768]
#out.pooler_output.shape   # [batch_size, 768]


class Cls_task(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self):
        super(Cls_task, self).__init__()  # 继承 __init__()

        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, token_type_ids, attention_mask):  # 正向传播
        with torch.no_grad():
            out = pre_trained(input_ids=input_ids,
                              token_type_ids=token_type_ids,
                              attention_mask=attention_mask)
        a = self.fc(out.pooler_output)  # a.shape: [batch_size, 2]
        a = a.softmax(dim=1)

        return a


classification_model = Cls_task()


# out_ = classification_model(input_ids=inputs,
#                            token_type_ids=token_type,
#                            attention_mask=attention_mask)


# step6: 训练
def train(model):
    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 定义学习率调节器
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_training_steps=len(train_data_loader),
                                                num_warmup_steps=0)
    best_train_accuracy = 0
    output_dir = './model'
    # 将模型切换到训练模式
    model.train()
    for i, (input_ids, token_type_ids, attention_mask, labels) in enumerate(train_data_loader):
        out_ = model(input_ids=input_ids,
                     token_type_ids=token_type_ids,
                     attention_mask=attention_mask)
        labels = torch.tensor(labels)
        loss = criterion(out_, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if i == 2:
            out = out_.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)
            if accuracy > best_train_accuracy:
                best_train_accuracy = accuracy
                output_model_file = os.path.join(output_dir, "checkpoint_{}.pt".format(i))
                checkpoint = {'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'steps': i,
                              'loss': loss}
                torch.save(checkpoint, output_model_file)

        if i >= 3:
            output_model_file = os.path.join(output_dir, "checkpoint_{}.pt".format(2))
            model = Cls_task()
            checkpoint = torch.load(output_model_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            out_ = model(input_ids=input_ids,
                         token_type_ids=token_type_ids,
                         attention_mask=attention_mask)
            labels = torch.tensor(labels)
            loss = criterion(out_, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if i % 2 == 0:
                out = out_.argmax(dim=1)
                accuracy = (out == labels).sum().item() / len(labels)
                if accuracy > best_train_accuracy:
                    best_train_accuracy = accuracy
                    checkpoint = {'model_state_dict': model.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict(),
                                  'steps': i,
                                  'loss': loss}
                    torch.save(checkpoint, output_model_file)
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                print(i, loss.item(), lr, accuracy)


if __name__ == '__main__':
    train(classification_model)
