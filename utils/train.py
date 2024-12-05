import plotly.graph_objects as go
import torch
from IPython.display import display, clear_output
from plotly.subplots import make_subplots
from sklearn.metrics import f1_score
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs, device,
                 enable_visualization=True, is_jupyter=True):
        """
        初始化训练器类
        :param model: 训练的模型
        :param train_dataloader: 训练数据加载器
        :param val_dataloader: 验证数据加载器
        :param optimizer: 优化器
        :param criterion: 损失函数
        :param num_epochs: 训练的轮数
        :param device: 设备（cpu/gpu）
        :param enable_visualization: 是否启用可视化 (默认启用)
        :param is_jupyter: 是否是 Jupyter 环境
        """
        self.device = device
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs

        self.best_f1 = 0
        self.f1_list = []
        self.loss_list = []
        self.enable_visualization = enable_visualization
        self.is_jupyter = is_jupyter

    def evaluate(self):
        self.model.eval()
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['images'].to(self.device)
                labels = batch['label'].to(self.device)

                # 预测
                outputs = self.model(input_ids, attention_mask, images)
                _, preds = torch.max(outputs, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average='weighted')
        return f1

    def update_visualization(self):
        fig = make_subplots(rows=1, cols=2, subplot_titles=('F1 Score', 'Loss'))

        fig.add_trace(
            go.Scatter(x=list(range(len(self.f1_list))), y=self.f1_list, mode='lines', name='F1 Score'),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=list(range(len(self.loss_list))), y=self.loss_list, mode='lines', name='Loss'),
            row=1, col=2
        )

        fig.update_layout(
            title_text="Training Progress",
            showlegend=True,
            xaxis_title="Epochs",
            yaxis_title="F1 Score",
            xaxis2_title="Batch",
            yaxis2_title="Loss",
            height=600,
            width=1800,
            template="plotly_white"
        )

        if self.is_jupyter:
            clear_output(wait=True)
            display(fig)

    def train(self):
        total_batches = self.num_epochs * len(self.train_dataloader)
        pbar = tqdm(total=total_batches, dynamic_ncols=True)

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            for i, batch in enumerate(self.train_dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['images'].to(self.device)
                labels = batch['label'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask, images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                avg_loss = total_loss / (i + 1)
                self.loss_list.append(avg_loss)

                if self.enable_visualization:
                    self.update_visualization()

                pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'F1': f'{self.f1_list[-1] if self.f1_list else 0:.4f}'})
                pbar.update(1)

            f1 = self.evaluate()
            self.f1_list.append(f1)

            torch.save(self.model.state_dict(), './save/last_model.pth')

            if f1 > self.best_f1:
                self.best_f1 = f1
                torch.save(self.model.state_dict(), './save/best_model.pth')

        pbar.close()
