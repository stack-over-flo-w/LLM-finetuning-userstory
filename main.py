import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# 读取数据并提取文本和标签
def read_data():
    file_path = 'user_stories_score.csv'  # 请将此路径替换为您的CSV文件的实际路径
    df = pd.read_csv(file_path)

    # 提取第一列和第二列的数据
    first_column_data = df.iloc[:, 0].tolist()
    second_column_data = df.iloc[:, 1].tolist()

    # 将标签转换为整数
    labels = []
    for item in second_column_data:
        if item == "Functional":
            labels.append(0)
        elif item == "Usability":
            labels.append(1)
        elif item == "Maintainability":
            labels.append(2)
        elif item == "Compatibility":
            labels.append(3)
        elif item == "Security":
            labels.append(4)
        elif item == "Performance":
            labels.append(5)
        elif item == "Reliability":
            labels.append(6)
        else:
            labels.append(7)

    return first_column_data, labels


def prepare_data(path,category):
    df = pd.read_csv(path)
    label = []
    # 提取第一列和第二列的数据
    first_column_data = df.iloc[:, 0].tolist()
    for i in range(len(first_column_data)):
        label.append(category)
    return first_column_data,label
# 加载并准备数据
texts, labels = read_data()

new_data1,new_label1 = prepare_data("more_data_llama/Functional.csv",0)
new_data2,new_label2 = prepare_data("more_data_llama/Security.csv",4)
new_data3,new_label3 = prepare_data("more_data_llama/Usability.csv",1)
new_data4,new_label4 = prepare_data("more_data_llama/Compatibility.csv",3)
new_data5,new_label5 = prepare_data("more_data_llama/Reliability.csv",6)
new_data6,new_label6 = prepare_data("more_data_llama/Maintainability.csv",2)
new_data7,new_label7 = prepare_data("more_data_llama/Portability.csv",7)
new_data8,new_label8 = prepare_data("more_data_llama/Performance.csv",5)

texts = texts + new_data3 + new_data4 + new_data5 + new_data5 + new_data6+ new_data7 + new_data8
labels = labels + new_label3 + new_label4 + new_label5  + new_label5 + new_label6+ new_label7 + new_label8

# 将数据拆分为训练集和测试集
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

#train_texts = train_texts + new_data3 + new_data4 + new_data5 + new_data5 + new_data6+ new_data7 + new_data8
#train_labels = train_labels + new_label3 + new_label4 + new_label5  + new_label5 + new_label6+ new_label7 + new_label8

# 加载预训练的 BERT 模型和 tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

# 创建数据集和数据加载器
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 创建数据集实例
train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=100)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_len=100)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=15, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=15)

# 自定义模型
class BertClassifier(torch.nn.Module):
    def __init__(self, bert_model, num_labels):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # 使用BERT的池化输出
        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)
        return logits

# 加载预训练的 BERT 模型
bert_model = BertModel.from_pretrained(model_name)

# 定义自定义模型
model = BertClassifier(bert_model, num_labels=8)

# 定义优化器和损失函数
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay =0.0001)
criterion = torch.nn.CrossEntropyLoss()

# 训练模型
def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# 评估模型
def eval_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).flatten()
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return total_loss / len(data_loader), accuracy_score(true_labels, predictions)

# 训练和评估模型
epochs = 50
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = eval_model(model, val_loader, criterion, device)
    print(f'Epoch {epoch + 1}/{epochs}')
    print(f'Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}')

# 最终评估模型
val_loss, val_acc = eval_model(model, val_loader, criterion, device)
print(f'Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}')
