import argparse
import torch
import multiprocessing
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from utils import data_partition, WarpSampler, evaluate
from model import SASRec


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def plot_and_save(loss_history, val_epochs, val_hr5, val_ndcg5, val_hr10, val_ndcg10, save_path, title='SASRec'):

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'SASRec Training Curves - {title}', fontsize=14, fontweight='bold')

    # 子图1：Loss 曲线
    ax1 = axes[0]
    ax1.plot(range(1, len(loss_history) + 1), loss_history, color='steelblue', linewidth=1.2, alpha=0.8)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 子图2：HR@5 和 HR@10 曲线
    ax2 = axes[1]
    ax2.plot(val_epochs, val_hr5,  marker='o', markersize=4, label='HR@5',  color='coral')
    ax2.plot(val_epochs, val_hr10, marker='s', markersize=4, label='HR@10', color='tomato')
    ax2.set_title('Hit Rate (Validation)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('HR')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 子图3：NDCG@5 和 NDCG@10 曲线
    ax3 = axes[2]
    ax3.plot(val_epochs, val_ndcg5,  marker='o', markersize=4, label='NDCG@5',  color='mediumseagreen')
    ax3.plot(val_epochs, val_ndcg10, marker='s', markersize=4, label='NDCG@10', color='seagreen')
    ax3.set_title('NDCG (Validation)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('NDCG')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n训练曲线图已保存至 {save_path}")


if __name__ == '__main__':
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=False, default='MOOC')
    parser.add_argument('--train_dir', required=False, default='default')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=201, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # 1. 加载数据
    dataset = data_partition(args.data)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size

    # 2. 初始化采样器和模型
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args).to(args.device)

    # 3. 定义损失函数与优化器
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # 设置保存路径
    if not os.path.exists('weights'):
        os.makedirs('weights')
    model_save_path = f"weights/best_sasrec_{args.data}.pth"

    best_val_ndcg = 0.0
    best_epoch = 0

    # 用于绘图的记录列表
    loss_history = []
    val_epochs  = []
    val_hr5     = []
    val_ndcg5   = []
    val_hr10    = []
    val_ndcg10  = []

    # 4. 训练循环
    print("Start Training...")
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)

            pos_logits, neg_logits = model(u, seq, pos, neg)

            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], torch.ones_like(pos_logits)[indices]) + \
                   bce_criterion(neg_logits[indices], torch.zeros_like(neg_logits)[indices])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = loss.item()
        loss_history.append(epoch_loss)
        print(f'Epoch {epoch:03d} | Loss: {epoch_loss:.4f}')

        # 5. 定期在验证集上评估
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_metrics = evaluate(model, dataset, args, is_test=False)
                print(f"Epoch {epoch} Valid - HR@5: {val_metrics['HR@5']:.4f}, NDCG@5: {val_metrics['NDCG@5']:.4f}, "
                      f"HR@10: {val_metrics['HR@10']:.4f}, NDCG@10: {val_metrics['NDCG@10']:.4f}")

                # 记录验证指标
                val_epochs.append(epoch)
                val_hr5.append(val_metrics['HR@5'])
                val_ndcg5.append(val_metrics['NDCG@5'])
                val_hr10.append(val_metrics['HR@10'])
                val_ndcg10.append(val_metrics['NDCG@10'])

                if val_metrics['NDCG@10'] > best_val_ndcg:
                    best_val_ndcg = val_metrics['NDCG@10']
                    best_epoch = epoch
                    torch.save(model.state_dict(), model_save_path)
                    print(f"--> [Saved] 找到新的最佳模型权重并保存至 {model_save_path}")

    sampler.close()

    # 6. 绘制并保存训练曲线图
    plot_and_save(
        loss_history, val_epochs,
        val_hr5, val_ndcg5, val_hr10, val_ndcg10,
        save_path=f"weights/training_curves_{args.data}.png",
        title = args.data
    )

    # 7. 最终测试阶段
    print(f"\nTraining finished! 加载最佳轮次 (Epoch {best_epoch}) 的模型进行最终测试...")

    best_model = SASRec(usernum, itemnum, args).to(args.device)
    best_model.load_state_dict(torch.load(model_save_path))
    best_model.eval()

    with torch.no_grad():
        test_metrics = evaluate(best_model, dataset, args, is_test=True)
        print("\n================ Final Test Results ================")
        print(f"HR@5:   {test_metrics['HR@5']:.4f}")
        print(f"NDCG@5: {test_metrics['NDCG@5']:.4f}")
        print(f"HR@10:  {test_metrics['HR@10']:.4f}")
        print(f"NDCG@10:{test_metrics['NDCG@10']:.4f}")
        print("====================================================\n")