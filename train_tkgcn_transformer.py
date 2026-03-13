import torch
import torch.nn as nn
import torch.optim as optim
from nets.TKGCN_V9_m import TKGCN_V9_m
import os

from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


# 读取文件，从数据集的文件夹中读取
def read_file(dir_path='TKG_dataset/train/tkgcn_dataset'):
    node_feature_s = []
    edge_index_s = []
    edge_attr_s = []
    dangerous_s = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)  # 得到每个文件夹的位置

        if os.path.isfile(file_path):
            print(f'Reading contents of file: {file_name}')

            with open(file_path, 'r') as file:
                content = file.read()
                split_content = content.split('\n\n')  # 划分三个矩阵

                node_feature_string = split_content[0].split('\n')[1:]
                edge_index_string = split_content[1].split('\n')[2:]
                edge_attr_string = split_content[2].split('\n')[2:]
                dangerous_string = split_content[3].split('\n')[2:-1]

                node_feature = []
                edge_index = []
                edge_attr = []
                dangerous = []

                for line in node_feature_string:  # 读取节点矩阵
                    line_value_list = []
                    for i in line.split(' '):
                        line_value_list.append(eval(i))
                    node_feature.append(line_value_list)
                node_feature_s.append(torch.tensor(node_feature, dtype=torch.float))

                line_value_list_1 = []
                line_value_list_2 = []
                for line in edge_index_string:  # 读取边的链接矩阵
                    line_s = line.split(' ')
                    line_value_list_1.append(eval(line_s[0]))
                    line_value_list_2.append(eval(line_s[1]))
                edge_index.append(line_value_list_1)
                edge_index.append(line_value_list_2)
                edge_index_s.append(torch.tensor(edge_index, dtype=torch.long))

                for line in edge_attr_string:  # 读取边的属性类别
                    edge_attr.append(eval(line.split('.')[0]))
                edge_attr_s.append(torch.tensor(edge_attr, dtype=torch.long))

                for line in dangerous_string:  # 读取危险类别
                    dangerous.append(eval(line))

                if len(dangerous) == 0:
                    print(f'Empty dangerous tensor in file {file_name}, skipping...')
                    print("####################################################################\n")
                    print("\n")
                    continue

                dangerous_s.append(torch.tensor(dangerous, dtype=torch.float))
                # 打印每个列表的长度以排查问题
                # print(f"node_feature_s length: {len(node_feature)}")
                # print(f"edge_index_s length: {len(edge_index)}")
                # print(f"edge_attr_s length: {len(edge_attr)}")
                # print(f"dangerous length: {len(dangerous)}")

    return node_feature_s, edge_index_s, edge_attr_s, dangerous_s


class myDataset(Dataset):
    def __init__(self, data_path='TKG_dataset/train/tkgcn_dataset'):
        self.node_feature_s = []
        self.edge_index_s = []
        self.edge_attr_s = []
        self.dangerous = []
        self.node_feature_s, self.edge_index_s, self.edge_attr_s, self.dangerous = read_file(data_path)

        # 打印每个列表的长度以排查问题
        print(f"node_feature_s length: {len(self.node_feature_s)}")
        print(f"edge_index_s length: {len(self.edge_index_s)}")
        print(f"edge_attr_s length: {len(self.edge_attr_s)}")
        print(f"dangerous length: {len(self.dangerous)}")

        # 检查数据长度是否一致，确保所有列表长度相同
        assert len(self.node_feature_s) == len(self.edge_index_s) == len(self.edge_attr_s) == len(self.dangerous), \
            "Lengths of node features, edge indices, edge attributes, and dangerous tensors do not match!"

    def __len__(self):
        return len(self.node_feature_s)

    def __getitem__(self, idx):
        # 增加边界检查，防止索引超出范围
        if idx >= len(self.node_feature_s):
            print(f"Index {idx} out of range for dataset length {len(self.node_feature_s)}")
            raise IndexError("Index out of range")

        return self.node_feature_s[idx], self.edge_index_s[idx], self.edge_attr_s[idx], self.dangerous[idx]


def evaluate_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for node_feature, edge_index, edge_attr, dangerous in tqdm(dataloader, desc='Evaluating'):
            node_feature = node_feature.to(device).squeeze()
            edge_index = edge_index.to(device).squeeze()
            edge_attr = edge_attr.to(device).squeeze()
            dangerous = dangerous.to(device).squeeze()

            if dangerous.numel() == 0:
                print(f'Empty dangerous tensor, skipping this sample')  # 有些数据文件打标签没打上
                continue

            outputs= model(node_feature, edge_index, edge_attr)

            try:
                max_index = torch.argmax(dangerous)
                predicted_index = torch.argmax(outputs)
                if predicted_index == max_index:
                    correct_predictions += 1
                total_predictions += 1
            except IndexError:
                print(f'Error in torch.argmax, dangerous shape: {dangerous.shape}')

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy


if __name__ == '__main__':
    device = torch.device('cuda')
    print(f'Using device: {device}')

    dataset = myDataset()  # 读取文件，生成数据集

    # Split the dataset         划分数据集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    tkgcn = TKGCN_V9_m(nhid=64,pooling_rate=0.001,dropout_rate=0.01,num_classes=2,num_features=60,num_relations=75).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(tkgcn.parameters(), lr=0.0001)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 35], gamma=0.1)

    best_accuracy = 0.0
    best_train_accuracy = 0.0
    accuracy_file = open('nets/nets_result/accuracy_of_tkgcn_transformer_4_14-v9-m.txt', 'w')  # 打开一个文件来记录准确度

    accuracy_file.write('TKGCN:\n')

    for epoch in range(30):
        tkgcn.train()  # Set the model to training mode
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0

        # 打印当前 epoch 的学习率
        # for param_group in optimizer.param_groups:
        #     print(f"Epoch {epoch + 1}, Learning Rate: {param_group['lr']}")

        for batch_idx, (node_feature, edge_index, edge_attr, dangerous) in enumerate(
                tqdm(train_loader, desc=f'Epoch {epoch + 1}')):
            node_feature = node_feature.to(device).squeeze()
            edge_index = edge_index.to(device).squeeze()
            edge_attr = edge_attr.to(device).squeeze()
            dangerous = dangerous.to(device).long().squeeze()


            if dangerous.numel() == 0:
                print(f'Empty dangerous tensor, skipping this sample')  # 有些数据文件打标签没打上
                continue

            outputs = tkgcn(node_feature, edge_index, edge_attr)

            outputs = outputs.unsqueeze(0) if outputs.dim() == 1 else outputs

            # Compute loss
            loss = criterion(outputs, dangerous.unsqueeze(0) if dangerous.dim() == 0 else dangerous)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate accuracy
            predicted = torch.argmax(outputs, dim=1)  # Predicted class
            correct_predictions += (predicted == dangerous).sum().item()
            total_predictions += dangerous.numel()

        # Calculate training accuracy
        train_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        # Save the best model based on test accuracy
        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
            torch.save(tkgcn.state_dict(), 'nets/nets_weights/best_tkgcn_train_model_414_m.pth')
            print("Model saved successfully!")


        # Evaluate model on test set
        tkgcn.eval()
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for node_feature, edge_index, edge_attr, dangerous in tqdm(test_loader, desc='Evaluating'):
                node_feature = node_feature.to(device).squeeze()
                edge_index = edge_index.to(device).squeeze()
                edge_attr = edge_attr.to(device).squeeze()
                dangerous = dangerous.to(device).long().squeeze()

                if dangerous.numel() == 0:
                    continue

                outputs = tkgcn(node_feature, edge_index, edge_attr)
                outputs = outputs.unsqueeze(0) if outputs.dim() == 1 else outputs
                predicted = torch.argmax(outputs, dim=1)
                correct_predictions += (predicted == dangerous).sum().item()
                total_predictions += dangerous.numel()

        test_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        # scheduler.step()

        # Save the best model based on test accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(tkgcn.state_dict(), 'nets/nets_weights/best_tkgcn_model_V9_414_m.pth')
            print("Model saved successfully!")

        # Log progress
        print(
            f'Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.4f}, Training Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
        accuracy_file.write(
            f'Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.4f}, Training Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}\n')

    accuracy_file.close()
    print(f'Training complete. Best Test Accuracy: {best_accuracy:.4f}')



