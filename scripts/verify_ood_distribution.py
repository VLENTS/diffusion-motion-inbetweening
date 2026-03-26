"""
验证「目标动作序列是否在先验分布外」的诊断脚本。

提供三种互补的验证方法：
1. t-SNE / UMAP 可视化：将训练集 & 目标序列映射到 2D，直观观察分布关系
2. 逐样本重建损失：用 diffusion 的 training_losses 对目标序列计算 per-sample loss
3. 特征空间统计距离：在 motion embedding 空间计算马氏距离 / k-NN 距离

用法：
    python scripts/verify_ood_distribution.py \
        --model_path save/condmdi/model000500000.pt \
        --target_motion_path /path/to/target_motions.npy \
        --output_dir save/ood_analysis
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from data_loaders.get_data import DatasetConfig, get_dataset_loader


# ============================================================
# 方法 1: Motion Embedding 提取 + t-SNE / UMAP 可视化
# ============================================================

def extract_motion_embeddings_from_mdm(model, motions, diffusion, t_probe=50):
    """
    通过在固定时间步 t_probe 对 x_0 加噪后送入模型，
    提取 Transformer encoder 输出作为 motion embedding。

    Args:
        model: MDM 模型
        motions: [N, njoints, nfeats, nframes] 归一化后的动作张量
        diffusion: GaussianDiffusion 实例
        t_probe: 用于加噪的 diffusion timestep（越小噪声越少，特征越清晰）

    Returns:
        embeddings: [N, latent_dim] 每条动作的全局 embedding
    """
    model.eval()
    device = next(model.parameters()).device
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(motions), 32):
            batch = motions[i:i+32].to(device)
            bs = batch.shape[0]
            nframes = batch.shape[-1]

            t = torch.full((bs,), t_probe, device=device, dtype=torch.long)
            noise = torch.randn_like(batch)
            x_t = diffusion.q_sample(batch, t, noise=noise)

            emb = model.embed_timestep(t)

            # encode text as empty (unconditioned)
            if 'text' in model.cond_mode:
                enc_text = model.encode_text([''] * bs)
                emb = emb + model.embed_text(enc_text)

            x_proc = model.input_process(x_t)  # [nframes, bs, latent_dim]

            xseq = torch.cat((emb, x_proc), axis=0)  # [nframes+1, bs, latent_dim]
            xseq = model.sequence_pos_encoder(xseq)
            output = model.seqTransEncoder(xseq)  # [nframes+1, bs, latent_dim]

            # 取所有帧的均值作为全局表征
            frame_features = output[1:]  # [nframes, bs, latent_dim]，去掉时间步 token
            global_emb = frame_features.mean(dim=0)  # [bs, latent_dim]
            embeddings.append(global_emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def visualize_tsne(train_emb, target_emb, output_path, method='tsne'):
    """
    用 t-SNE 或 UMAP 可视化两组 embedding 的分布关系。
    """
    all_emb = np.concatenate([train_emb, target_emb], axis=0)
    labels = np.array([0] * len(train_emb) + [1] * len(target_emb))

    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=min(30, len(all_emb) - 1),
                       random_state=42, n_iter=1000)
        coords = reducer.fit_transform(all_emb)
    elif method == 'umap':
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(all_emb)
    else:
        raise ValueError(f"Unknown method: {method}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    train_mask = labels == 0
    target_mask = labels == 1

    ax.scatter(coords[train_mask, 0], coords[train_mask, 1],
               c='#4A90D9', alpha=0.3, s=10, label=f'Train set (n={train_mask.sum()})')
    ax.scatter(coords[target_mask, 0], coords[target_mask, 1],
               c='#E74C3C', alpha=0.8, s=40, marker='*', label=f'Target HSI (n={target_mask.sum()})')

    ax.set_title(f'Motion Distribution ({method.upper()})', fontsize=14)
    ax.legend(fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 已保存至 {output_path}")


# ============================================================
# 方法 2: 逐样本 Diffusion 重建损失
# ============================================================

def compute_per_sample_loss(model, diffusion, motions, model_kwargs, n_timesteps=10):
    """
    对每个样本在多个 diffusion timestep 上计算重建损失，取均值。
    OOD 样本的重建损失通常显著高于训练分布内的样本。

    Args:
        model: MDM 模型
        diffusion: GaussianDiffusion 实例
        motions: [N, njoints, nfeats, nframes] 归一化动作
        model_kwargs: 包含 y['mask'], y['lengths'] 等信息的 dict
        n_timesteps: 采样的时间步数量

    Returns:
        per_sample_loss: [N] 每个样本的平均重建损失
    """
    model.eval()
    device = next(model.parameters()).device
    N = len(motions)

    timesteps_to_probe = np.linspace(0, diffusion.num_timesteps - 1, n_timesteps, dtype=int)
    all_losses = torch.zeros(N, device='cpu')
    count = 0

    with torch.no_grad():
        for t_val in tqdm(timesteps_to_probe, desc="Computing reconstruction loss"):
            for i in range(0, N, 32):
                batch = motions[i:i+32].to(device)
                bs = batch.shape[0]
                t = torch.full((bs,), t_val, device=device, dtype=torch.long)

                batch_kwargs = {
                    'y': {
                        'mask': model_kwargs['y']['mask'][i:i+32].to(device),
                        'lengths': model_kwargs['y']['lengths'][i:i+32],
                        'text': model_kwargs['y']['text'][i:i+32] if 'text' in model_kwargs['y'] else [''] * bs,
                    }
                }
                if 'obs_x0' in model_kwargs:
                    batch_kwargs['obs_x0'] = batch
                    batch_kwargs['obs_mask'] = torch.zeros_like(batch, dtype=torch.bool)

                terms = diffusion.training_losses(model, batch, t, model_kwargs=batch_kwargs)
                all_losses[i:i+bs] += terms['loss'].cpu()
            count += 1

    return all_losses / count


def plot_loss_histogram(train_losses, target_losses, output_path):
    """
    绘制训练集 vs 目标序列的 loss 分布直方图。
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.hist(train_losses, bins=50, alpha=0.6, color='#4A90D9',
            label=f'Train set (mean={train_losses.mean():.4f})', density=True)
    ax.hist(target_losses, bins=50, alpha=0.6, color='#E74C3C',
            label=f'Target HSI (mean={target_losses.mean():.4f})', density=True)

    for tl in target_losses:
        ax.axvline(x=tl, color='#E74C3C', linestyle='--', alpha=0.5, linewidth=0.8)

    ax.set_xlabel('Reconstruction Loss', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Per-sample Reconstruction Loss Distribution', fontsize=14)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Loss 分布] 已保存至 {output_path}")


# ============================================================
# 方法 3: 特征空间统计距离
# ============================================================

def compute_distributional_metrics(train_emb, target_emb):
    """
    在 embedding 空间计算统计距离指标。

    Returns:
        dict: 包含各项距离度量
    """
    from scipy.spatial.distance import mahalanobis
    from scipy.linalg import inv

    train_mean = train_emb.mean(axis=0)
    train_cov = np.cov(train_emb.T)

    # 正则化协方差矩阵
    reg = 1e-6 * np.eye(train_cov.shape[0])
    train_cov_inv = inv(train_cov + reg)

    # 每个目标样本到训练分布的马氏距离
    mahal_distances = []
    for i in range(len(target_emb)):
        d = mahalanobis(target_emb[i], train_mean, train_cov_inv)
        mahal_distances.append(d)
    mahal_distances = np.array(mahal_distances)

    # 训练集内样本到自身分布的马氏距离（用于对比）
    train_mahal = []
    for i in range(min(500, len(train_emb))):
        d = mahalanobis(train_emb[i], train_mean, train_cov_inv)
        train_mahal.append(d)
    train_mahal = np.array(train_mahal)

    # k-NN 距离: 每个目标样本到训练集最近 k 个样本的平均距离
    from sklearn.neighbors import NearestNeighbors
    k = min(5, len(train_emb))
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(train_emb)
    target_knn_dist, _ = nn.kneighbors(target_emb)
    target_knn_mean = target_knn_dist.mean(axis=1)

    train_knn_dist, _ = nn.kneighbors(train_emb[:min(500, len(train_emb))])
    train_knn_mean = train_knn_dist.mean(axis=1)

    return {
        'target_mahal': mahal_distances,
        'train_mahal': train_mahal,
        'target_knn': target_knn_mean,
        'train_knn': train_knn_mean,
    }


def plot_distance_comparison(metrics, output_path):
    """
    绘制马氏距离和 k-NN 距离的对比图。
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 马氏距离
    ax = axes[0]
    ax.hist(metrics['train_mahal'], bins=40, alpha=0.6, color='#4A90D9',
            label=f"Train (mean={metrics['train_mahal'].mean():.2f})", density=True)
    for d in metrics['target_mahal']:
        ax.axvline(x=d, color='#E74C3C', linestyle='--', alpha=0.7, linewidth=1.2)
    ax.axvline(x=metrics['target_mahal'].mean(), color='#E74C3C', linewidth=2.5,
               label=f"Target mean={metrics['target_mahal'].mean():.2f}")
    ax.set_xlabel('Mahalanobis Distance', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Mahalanobis Distance to Train Distribution', fontsize=13)
    ax.legend(fontsize=11)

    # k-NN 距离
    ax = axes[1]
    ax.hist(metrics['train_knn'], bins=40, alpha=0.6, color='#4A90D9',
            label=f"Train (mean={metrics['train_knn'].mean():.2f})", density=True)
    for d in metrics['target_knn']:
        ax.axvline(x=d, color='#E74C3C', linestyle='--', alpha=0.7, linewidth=1.2)
    ax.axvline(x=metrics['target_knn'].mean(), color='#E74C3C', linewidth=2.5,
               label=f"Target mean={metrics['target_knn'].mean():.2f}")
    ax.set_xlabel('k-NN Distance', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('k-NN Distance to Train Set', fontsize=13)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[统计距离] 已保存至 {output_path}")


# ============================================================
# 主流程
# ============================================================

def load_train_data(args, max_frames=196, n_samples=500):
    """加载训练集子集用于分布对比。"""
    conf = DatasetConfig(
        name=args.dataset,
        batch_size=min(n_samples, 64),
        num_frames=max_frames,
        split='train',
        hml_mode='train',
        use_abs3d=args.abs_3d,
        traject_only=args.traj_only,
        use_random_projection=args.use_random_proj,
        random_projection_scale=args.random_proj_scale,
        augment_type='none',
        std_scale_shift=args.std_scale_shift,
        drop_redundant=args.drop_redundant,
    )
    data = get_dataset_loader(conf)
    return data


def main():
    parser = argparse.ArgumentParser(description='OOD 分布验证')
    parser.add_argument('--model_path', type=str, required=True,
                        help='CondMDI checkpoint 路径')
    parser.add_argument('--target_motion_path', type=str, default=None,
                        help='目标动作 .npy 文件路径 [N, njoints, nfeats, nframes]（已归一化）')
    parser.add_argument('--output_dir', type=str, default='save/ood_analysis')
    parser.add_argument('--n_train_samples', type=int, default=500,
                        help='从训练集采样的数量')
    parser.add_argument('--t_probe', type=int, default=50,
                        help='embedding 提取时使用的 diffusion timestep')
    parser.add_argument('--vis_method', type=str, default='tsne',
                        choices=['tsne', 'umap'])
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- 加载模型 ----------
    # 借用 cond_synt_args 从 checkpoint 自动恢复训练参数
    from utils.parser_util import cond_synt_args
    sys.argv = [sys.argv[0],
                '--model_path', args.model_path,
                '--num_samples', '1',
                '--edit_mode', 'benchmark_sparse']
    model_args = cond_synt_args(model_path=args.model_path)
    model_args.device = args.device
    dist_util.setup_dist(args.device)

    max_frames = 196 if model_args.dataset in ['kit', 'humanml'] else 200

    print("加载训练数据...")
    train_loader = load_train_data(model_args, max_frames, args.n_train_samples)

    print("创建模型和扩散过程...")
    model, diffusion = create_model_and_diffusion(model_args, train_loader)
    load_saved_model(model, args.model_path)
    model.to(args.device)
    model.eval()

    # ---------- 收集训练集样本 ----------
    print(f"收集训练集样本 (目标 {args.n_train_samples} 条)...")
    train_motions = []
    train_kwargs_list = {'mask': [], 'lengths': [], 'text': []}
    for batch_i, (motion, kwargs) in enumerate(train_loader):
        train_motions.append(motion)
        train_kwargs_list['mask'].append(kwargs['y']['mask'])
        train_kwargs_list['lengths'].append(kwargs['y']['lengths'])
        train_kwargs_list['text'].extend(kwargs['y']['text'])
        if sum(m.shape[0] for m in train_motions) >= args.n_train_samples:
            break
    train_motions = torch.cat(train_motions, dim=0)[:args.n_train_samples]
    train_kwargs = {
        'y': {
            'mask': torch.cat(train_kwargs_list['mask'], dim=0)[:args.n_train_samples],
            'lengths': torch.cat(train_kwargs_list['lengths'], dim=0)[:args.n_train_samples],
            'text': train_kwargs_list['text'][:args.n_train_samples],
        }
    }

    # ---------- 加载目标动作 ----------
    if args.target_motion_path is not None:
        print(f"加载目标动作: {args.target_motion_path}")
        target_data = np.load(args.target_motion_path, allow_pickle=True)
        if isinstance(target_data, np.ndarray) and target_data.dtype == object:
            target_data = target_data.item()
        if isinstance(target_data, dict):
            target_motions = torch.tensor(target_data['motion']).float()
        else:
            target_motions = torch.tensor(target_data).float()
        if target_motions.ndim == 3:
            target_motions = target_motions.unsqueeze(2)
    else:
        # 如果没有提供目标动作，用测试集作为演示
        print("未提供 --target_motion_path，使用测试集作为演示...")
        test_conf = DatasetConfig(
            name=model_args.dataset,
            batch_size=64,
            num_frames=max_frames,
            split='test',
            hml_mode='train',
            use_abs3d=model_args.abs_3d,
            traject_only=model_args.traj_only,
            use_random_projection=model_args.use_random_proj,
            random_projection_scale=model_args.random_proj_scale,
            augment_type='none',
            std_scale_shift=model_args.std_scale_shift,
            drop_redundant=model_args.drop_redundant,
        )
        test_loader = get_dataset_loader(test_conf)
        test_batch = next(iter(test_loader))
        target_motions = test_batch[0]

    n_target = len(target_motions)
    print(f"训练集: {len(train_motions)} 条, 目标序列: {n_target} 条")
    print(f"动作形状: {train_motions.shape}")

    # ========== 方法 1: Embedding 可视化 ==========
    print("\n" + "="*60)
    print("方法 1: Transformer Embedding 提取 + 可视化")
    print("="*60)

    train_emb = extract_motion_embeddings_from_mdm(
        model, train_motions, diffusion, t_probe=args.t_probe)
    target_emb = extract_motion_embeddings_from_mdm(
        model, target_motions, diffusion, t_probe=args.t_probe)

    visualize_tsne(train_emb, target_emb,
                   os.path.join(args.output_dir, f'distribution_{args.vis_method}.png'),
                   method=args.vis_method)

    # ========== 方法 2: 逐样本重建损失 ==========
    print("\n" + "="*60)
    print("方法 2: 逐样本 Diffusion 重建损失")
    print("="*60)

    train_losses = compute_per_sample_loss(
        model, diffusion, train_motions, train_kwargs, n_timesteps=10)

    target_kwargs = {
        'y': {
            'mask': torch.ones(n_target, 1, 1, max_frames),
            'lengths': torch.full((n_target,), max_frames, dtype=torch.long),
            'text': [''] * n_target,
        }
    }
    target_losses = compute_per_sample_loss(
        model, diffusion, target_motions, target_kwargs, n_timesteps=10)

    train_losses_np = train_losses.numpy()
    target_losses_np = target_losses.numpy()

    plot_loss_histogram(train_losses_np, target_losses_np,
                        os.path.join(args.output_dir, 'loss_distribution.png'))

    # ========== 方法 3: 统计距离 ==========
    print("\n" + "="*60)
    print("方法 3: 特征空间统计距离")
    print("="*60)

    metrics = compute_distributional_metrics(train_emb, target_emb)
    plot_distance_comparison(metrics, os.path.join(args.output_dir, 'distance_comparison.png'))

    # ========== 汇总报告 ==========
    print("\n" + "="*60)
    print("OOD 诊断报告")
    print("="*60)

    report = []
    report.append(f"训练集样本数: {len(train_motions)}")
    report.append(f"目标序列数:   {n_target}")
    report.append(f"")
    report.append(f"[重建损失]")
    report.append(f"  训练集均值: {train_losses_np.mean():.6f} ± {train_losses_np.std():.6f}")
    report.append(f"  目标均值:   {target_losses_np.mean():.6f} ± {target_losses_np.std():.6f}")
    report.append(f"  比值:       {target_losses_np.mean() / train_losses_np.mean():.2f}x")
    report.append(f"")
    report.append(f"[马氏距离]")
    report.append(f"  训练集到自身: {metrics['train_mahal'].mean():.4f} ± {metrics['train_mahal'].std():.4f}")
    report.append(f"  目标到训练集: {metrics['target_mahal'].mean():.4f} ± {metrics['target_mahal'].std():.4f}")
    report.append(f"  比值:         {metrics['target_mahal'].mean() / metrics['train_mahal'].mean():.2f}x")
    report.append(f"")
    report.append(f"[k-NN 距离]")
    report.append(f"  训练集内部: {metrics['train_knn'].mean():.4f} ± {metrics['train_knn'].std():.4f}")
    report.append(f"  目标到训练: {metrics['target_knn'].mean():.4f} ± {metrics['target_knn'].std():.4f}")
    report.append(f"  比值:       {metrics['target_knn'].mean() / metrics['train_knn'].mean():.2f}x")
    report.append(f"")
    report.append(f"[OOD 判定]")
    loss_ratio = target_losses_np.mean() / train_losses_np.mean()
    mahal_ratio = metrics['target_mahal'].mean() / metrics['train_mahal'].mean()
    knn_ratio = metrics['target_knn'].mean() / metrics['train_knn'].mean()

    if loss_ratio > 1.5 or mahal_ratio > 2.0 or knn_ratio > 2.0:
        report.append(f"  >>> 目标序列 **很可能在先验分布外** (OOD)")
        report.append(f"      loss比值={loss_ratio:.2f}x, 马氏距离比值={mahal_ratio:.2f}x, kNN比值={knn_ratio:.2f}x")
    elif loss_ratio > 1.2 or mahal_ratio > 1.5 or knn_ratio > 1.5:
        report.append(f"  >>> 目标序列 **处于分布边缘** (边界 OOD)")
        report.append(f"      loss比值={loss_ratio:.2f}x, 马氏距离比值={mahal_ratio:.2f}x, kNN比值={knn_ratio:.2f}x")
    else:
        report.append(f"  >>> 目标序列 **在分布内** (In-distribution)")
        report.append(f"      loss比值={loss_ratio:.2f}x, 马氏距离比值={mahal_ratio:.2f}x, kNN比值={knn_ratio:.2f}x")

    report_text = '\n'.join(report)
    print(report_text)

    report_path = os.path.join(args.output_dir, 'ood_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\n完整报告已保存至 {report_path}")

    # 保存原始数值
    np.savez(os.path.join(args.output_dir, 'raw_metrics.npz'),
             train_emb=train_emb,
             target_emb=target_emb,
             train_losses=train_losses_np,
             target_losses=target_losses_np,
             target_mahal=metrics['target_mahal'],
             train_mahal=metrics['train_mahal'],
             target_knn=metrics['target_knn'],
             train_knn=metrics['train_knn'])
    print(f"原始数值已保存至 {os.path.join(args.output_dir, 'raw_metrics.npz')}")


if __name__ == '__main__':
    main()
