import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from confusion_matrix import plot_confusion_matrix1
from torchvision import transforms
import dataset_loader
import os
from WCANet import wcanet50,wcanet101



def main():
    # 1. load dataset
    root = '/mnt/tree'
    loss_train=[]
    acc=[]
    filename1 = '/mnt/train_ACC.txt'
    filename2 = '/mnt/test_ACC.txt'
    filename3= '/mnt/avl.txt'
    batch_size = 8
    train_dataset = dataset_loader.mydata(root, train=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=False,num_workers=12)
    test_dataset = dataset_loader.mydata(root, train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,drop_last=False,num_workers=12)

    # 2. load model
    model=wcanet50(pretrained=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # 3. prepare super parameters
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch = 100
    # 4. train
    val_acc_list = []
    out_dir = "/mnt/gsop/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for epoch in range(0, epoch):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        pre = []
        true=[]
        for batch_idx, (images, labels) in enumerate(train_loader):
            length = len(train_loader)
            labels = torch.Tensor(labels.numpy())
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)  # torch.size([batch_size, num_class])
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            loss_train.append(sum_loss / (batch_idx + 1))
            #print(loss_train)
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (batch_idx + 1 + epoch * length), sum_loss / (batch_idx + 1), 100. * correct / total))
        with open(filename1, 'a') as f:  # 'a'表示append,即在原来文件内容后继续写数据（不清楚原有数据）
            f.write(str(epoch + 1)+"  Loss: " + str(sum_loss / (batch_idx + 1))+" Acc: "+ str(100. * correct / total))
            f.write('\r\n')
        # get the ac with testdataset in each epoch
        print('Waiting Val...')
        with torch.no_grad():
            correct = 0.0
            total = 0.0
            char=0.0
            for batch_idx, (images, labels) in enumerate(test_loader):
                model.eval()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, dim=1)
                pre.extend(predicted.cpu().numpy().tolist())
                true.extend(labels.cpu().numpy().tolist())
                total += labels.size(0)
                correct += (predicted == labels).sum()
                char=char+1
            print('Val\'s ac is: %.3f%%' % (100 * correct / total))

            acc_val = 100 * correct / total
            val_acc_list.append(acc_val)
            acc.append(acc_val.item())
            with open(filename2, 'a') as f:  # 'a'表示append,即在原来文件内容后继续写数据（不清楚原有数据）
                f.write(str(acc_val))
                f.write('\r\n')
        torch.save(model.state_dict(), out_dir + "last1.pt")
        if acc_val == max(val_acc_list):
            torch.save(model.state_dict(), out_dir + "best1.pt")
            print("save epoch {} model".format(epoch))
            cmtx = confusion_matrix(true, pre)
            print(cmtx)
            t = classification_report(true, pre, target_names=["C1","C2","C3","C4","C5","C6","C7","C8"])
            print(t)
            with open(filename3, 'w') as f:  # 'a'表示append,即在原来文件内容后继续写数据（不清楚原有数据）
                f.write(t)
                f.write('\r\n')
            label = ["C1","C2","C3","C4","C5","C6","C7","C8"]
            plot_confusion_matrix1(cmtx,label,epoch+1)

if __name__ == "__main__":
    main()
