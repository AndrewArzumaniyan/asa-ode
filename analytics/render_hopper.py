"""
Сравнение реальных и предсказанных траекторий Hopper.

Левое видео:  полная реальная траектория
Правое видео: первая половина реальная, вторая — предсказана моделью

Запуск:
    python analytics/render_hopper.py
    python analytics/render_hopper.py --idx 3 --save mp4
    python analytics/render_hopper.py --mode snapshots --n 6
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from pathlib import Path
from dm_control import suite


# ─── Загрузка данных ───────────────────────────────────────────────────────────

def load_data(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(data, torch.Tensor):
        arr = data.numpy()
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise ValueError(f'Неожиданный формат: {type(data)}')

    # Нормализация как в mujoco_physics.py: data_norm = (data - att_min) / att_max
    reshaped = arr.reshape(-1, arr.shape[-1])
    att_min  = reshaped.min(axis=0)
    att_max  = reshaped.max(axis=0)
    att_max[att_max == 0.] = 1.
    data_norm = (arr - att_min) / att_max

    return arr, data_norm, att_min, att_max


# ─── Загрузка модели ──────────────────────────────────────────────────────────

def load_model(ckpt_path, root, device):
    """Загружает Latent ODE модель из чекпоинта."""
    import sys
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from lib.create_latent_ode_model import create_LatentODE_model
    from torch.distributions import Normal

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt['args']

    obsrv_std = torch.Tensor([1e-3]).to(device)
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.0]).to(device))

    # input_dim для hopper = 14 (7 qpos + 7 qvel)
    input_dim = 14
    model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, args


# ─── Инференс ─────────────────────────────────────────────────────────────────

def predict_second_half(model, traj_gt_raw, att_min, att_max, device, obs_fraction=1/3):
    """
    Запускает модель на первой части траектории, возвращает предсказание остатка.

    traj_gt_raw: numpy (T, C) — сырые данные (qpos/qvel)
    Возвращает: pred_raw numpy (T_pred, C) — денормализованное предсказание
                split_t: int
    """
    T, C = traj_gt_raw.shape
    split_t = max(1, int(T * obs_fraction))

    # Нормализуем вход (как при обучении)
    traj_norm = (traj_gt_raw - att_min) / att_max

    time_steps = torch.linspace(0.0, 1.0, T, device=device)
    obs_data = torch.tensor(traj_norm[:split_t], dtype=torch.float32, device=device).unsqueeze(0)
    obs_mask = torch.ones(1, split_t, C, device=device)

    with torch.no_grad():
        pred, _ = model.get_reconstruction(
            time_steps_to_predict=time_steps[split_t:],
            truth=obs_data,
            truth_time_steps=time_steps[:split_t],
            mask=obs_mask,
            n_traj_samples=1,
            mode='extrap',
        )
    pred_norm = pred[0, 0].cpu().numpy()  # (T_pred, C)

    # Денормализуем обратно в пространство qpos/qvel
    pred_raw = pred_norm * att_max + att_min
    return pred_raw, split_t


# ─── Рендеринг одной траектории ───────────────────────────────────────────────

def _make_physics():
    env = suite.load('hopper', 'hop')
    env.reset()
    return env.physics


def render_frames(traj, physics=None, width=480, height=320):
    """traj: numpy (T, C). Возвращает list of RGB frames."""
    if physics is None:
        physics = _make_physics()
    n_qpos = physics.data.qpos.shape[0]
    n_qvel = physics.data.qvel.shape[0]
    frames = []
    for t in range(len(traj)):
        physics.data.qpos[:] = traj[t, :n_qpos]
        physics.data.qvel[:] = traj[t, n_qpos:n_qpos + n_qvel]
        physics.forward()
        frames.append(physics.render(height=height, width=width, camera_id=0))
    return frames


# ─── Side-by-side: GT vs Predicted ───────────────────────────────────────────

def show_gt_vs_pred(traj_gt, pred_second_half, split_t, idx,
                    fps=25, width=480, height=320, save=None):
    """
    Два видео рядом:
      Слева:  полная реальная траектория
      Справа: первая часть реальная, вторая — предсказана моделью
    """
    T = len(traj_gt)

    print('Рендеринг реальной траектории...')
    physics = _make_physics()
    frames_gt = render_frames(traj_gt, physics, width=width, height=height)

    # Собираем правую траекторию: первая часть из GT, вторая из модели
    traj_pred_full = np.concatenate([traj_gt[:split_t], pred_second_half], axis=0)
    print('Рендеринг предсказанной траектории...')
    frames_pred = render_frames(traj_pred_full, physics, width=width, height=height)

    # ── Рисуем ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('#0d0d0d')
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.02, right=0.98, wspace=0.06)

    for ax in axes:
        ax.axis('off')

    axes[0].set_title('Ground Truth', color='#dddddd', fontsize=13, pad=6)
    axes[1].set_title('Predicted (2nd half)', color='#dddddd', fontsize=13, pad=6)

    im_gt   = axes[0].imshow(frames_gt[0])
    im_pred = axes[1].imshow(frames_pred[0])

    # Цветная полоска-индикатор прогресса
    bar_ax = fig.add_axes([0.05, 0.06, 0.90, 0.04])
    bar_ax.set_xlim(0, T - 1)
    bar_ax.set_ylim(0, 1)
    bar_ax.axis('off')

    # Фон: серый — observed, оранжевый — predicted
    bar_ax.axvspan(0,         split_t - 1, ymin=0, ymax=1, color='#3a7ebf', alpha=0.55)
    bar_ax.axvspan(split_t-1, T - 1,       ymin=0, ymax=1, color='#e07b39', alpha=0.45)

    cursor_line, = bar_ax.plot([0, 0], [0, 1], color='white', lw=2, zorder=5)
    bar_ax.axvline(split_t - 1, color='white', lw=1.5, ls='--', alpha=0.8)

    obs_patch  = mpatches.Patch(color='#3a7ebf', label='Observed')
    pred_patch = mpatches.Patch(color='#e07b39', label='Predicted')
    bar_ax.legend(handles=[obs_patch, pred_patch],
                  loc='upper right', framealpha=0, fontsize=9,
                  labelcolor='white', bbox_to_anchor=(1.0, 2.8))

    # Временна́я метка
    time_text = fig.text(0.50, 0.01, '', ha='center', va='bottom',
                         color='#cccccc', fontsize=10)

    def update(t):
        im_gt.set_data(frames_gt[t])
        im_pred.set_data(frames_pred[t])
        cursor_line.set_xdata([t, t])

        phase = 'OBSERVED' if t < split_t else 'PREDICTED'
        color = '#7ab8e8' if t < split_t else '#e8a06a'
        time_text.set_text(f'frame {t:03d} / {T-1}   [{phase}]')
        time_text.set_color(color)
        return im_gt, im_pred, cursor_line, time_text

    ani = animation.FuncAnimation(fig, update, frames=T,
                                  interval=1000 // fps, blit=False)

    if save == 'gif':
        fname = f'hopper_comparison_{idx}.gif'
        ani.save(fname, writer='pillow', fps=fps)
        print(f'Сохранено: {fname}')
    elif save == 'mp4':
        fname = f'hopper_comparison_{idx}.mp4'
        ani.save(fname, writer='ffmpeg', fps=fps)
        print(f'Сохранено: {fname}')
    else:
        plt.show()


# ─── Последовательные траектории (10 подряд: 5 train + 5 test) ───────────────

def show_sequential(data, att_min, att_max, model, device, train_indices, test_indices,
                    fps=25, width=320, height=220, save=None):
    """
    Воспроизводит n_train+n_test траекторий подряд: сначала train, потом test.
    Каждая показывается как GT vs Predicted side-by-side.
    Всё — одна анимация / один файл.
    """
    physics = _make_physics()

    segments = []  # (label, split_label, frames_gt, frames_pred, split_t)
    groups = [('TRAIN', train_indices), ('TEST', test_indices)]

    for split_name, indices in groups:
        for rank, idx in enumerate(indices, 1):
            print(f'[{split_name} {rank}/{len(indices)}] инференс traj #{idx}...')
            traj_gt = data[idx]
            pred, split_t = predict_second_half(model, traj_gt, att_min, att_max, device)
            traj_pred_full = np.concatenate([traj_gt[:split_t], pred], axis=0)

            print(f'  рендеринг GT...')
            frames_gt   = render_frames(traj_gt,        physics, width=width, height=height)
            print(f'  рендеринг Pred...')
            frames_pred = render_frames(traj_pred_full, physics, width=width, height=height)

            label = f'{split_name}  {rank}/{len(indices)}   traj #{idx}'
            segments.append((label, split_name, frames_gt, frames_pred, split_t))

    T          = len(segments[0][2])
    n_seg      = len(segments)
    total_frames = T * n_seg

    TRAIN_CLR = '#4a9eff'
    TEST_CLR  = '#ff8c42'

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor('#0d0d0d')
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.02, right=0.98, wspace=0.05)

    for ax in axes:
        ax.axis('off')
    axes[0].set_title('Ground Truth',       color='#dddddd', fontsize=12, pad=5)
    axes[1].set_title('Predicted (2nd half)', color='#dddddd', fontsize=12, pad=5)

    im_gt   = axes[0].imshow(segments[0][2][0])
    im_pred = axes[1].imshow(segments[0][3][0])

    # Прогресс-бар
    bar_ax = fig.add_axes([0.05, 0.07, 0.90, 0.035])
    bar_ax.set_xlim(0, T - 1)
    bar_ax.set_ylim(0, 1)
    bar_ax.axis('off')
    bar_ax.axvspan(0, segments[0][4] - 1, color='#3a7ebf', alpha=0.5)
    bar_ax.axvspan(segments[0][4] - 1, T - 1, color='#e07b39', alpha=0.4)
    bar_ax.axvline(segments[0][4] - 1, color='white', lw=1.2, ls='--', alpha=0.7)
    cursor, = bar_ax.plot([0, 0], [0, 1], color='white', lw=2, zorder=5)

    # Общий прогресс
    total_bar_ax = fig.add_axes([0.05, 0.03, 0.90, 0.020])
    total_bar_ax.set_xlim(0, total_frames - 1)
    total_bar_ax.set_ylim(0, 1)
    total_bar_ax.axis('off')
    # Раскрасим секции train/test
    for si, (_, sname, _, _, _) in enumerate(segments):
        c = TRAIN_CLR if sname == 'TRAIN' else TEST_CLR
        total_bar_ax.axvspan(si * T, (si + 1) * T, color=c, alpha=0.35)
        if si > 0:
            total_bar_ax.axvline(si * T, color='white', lw=0.8, alpha=0.5)
    total_cursor, = total_bar_ax.plot([0, 0], [0, 1], color='white', lw=2, zorder=5)

    # Надписи
    seg_label   = fig.text(0.50, 0.93, '', ha='center', va='top',
                           color='white', fontsize=13, fontweight='bold')
    time_label  = fig.text(0.50, 0.01, '', ha='center', va='bottom',
                           color='#aaaaaa', fontsize=9)

    # Легенда для общей полоски
    train_p = mpatches.Patch(color=TRAIN_CLR, alpha=0.6, label='Train')
    test_p  = mpatches.Patch(color=TEST_CLR,  alpha=0.6, label='Test')
    total_bar_ax.legend(handles=[train_p, test_p], loc='upper right',
                        framealpha=0, fontsize=8, labelcolor='white',
                        bbox_to_anchor=(1.0, 5.0))

    _prev_seg = [-1]

    def update(global_t):
        seg_idx    = global_t // T
        local_t    = global_t  % T
        label, sname, frames_gt, frames_pred, split_t = segments[seg_idx]

        im_gt.set_data(frames_gt[local_t])
        im_pred.set_data(frames_pred[local_t])
        cursor.set_xdata([local_t, local_t])
        total_cursor.set_xdata([global_t, global_t])

        # Обновляем прогресс-бар только при смене сегмента
        if seg_idx != _prev_seg[0]:
            for coll in bar_ax.collections:
                coll.remove()
            bar_ax.axvspan(0, split_t - 1, color='#3a7ebf', alpha=0.5)
            bar_ax.axvspan(split_t - 1, T - 1, color='#e07b39', alpha=0.4)
            bar_ax.axvline(split_t - 1, color='white', lw=1.2, ls='--', alpha=0.7)
            clr = TRAIN_CLR if sname == 'TRAIN' else TEST_CLR
            seg_label.set_text(label)
            seg_label.set_color(clr)
            _prev_seg[0] = seg_idx

        phase = 'OBSERVED' if local_t < split_t else 'PREDICTED'
        phase_clr = '#7ab8e8' if local_t < split_t else '#e8a06a'
        time_label.set_text(f'frame {local_t:03d}/{T-1}   [{phase}]   '
                            f'({seg_idx+1}/{n_seg})')
        time_label.set_color(phase_clr)

        return im_gt, im_pred, cursor, total_cursor, seg_label, time_label

    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                  interval=1000 // fps, blit=False)

    if save == 'gif':
        fname = 'hopper_sequential.gif'
        ani.save(fname, writer='pillow', fps=fps)
        print(f'Сохранено: {fname}')
    elif save == 'mp4':
        fname = 'hopper_sequential.mp4'
        ani.save(fname, writer='ffmpeg', fps=fps)
        print(f'Сохранено: {fname}')
    else:
        plt.show()


# ─── Стоп-кадры GT vs Predicted ──────────────────────────────────────────────

def show_snapshots_gt_vs_pred(traj_gt, pred_second_half, split_t, idx,
                               n_snaps=6, width=320, height=220):
    T = len(traj_gt)
    traj_pred_full = np.concatenate([traj_gt[:split_t], pred_second_half], axis=0)

    ts = np.linspace(0, T - 1, n_snaps, dtype=int)

    physics = _make_physics()
    frames_gt   = render_frames(traj_gt,       physics, width=width, height=height)
    frames_pred = render_frames(traj_pred_full, physics, width=width, height=height)

    fig, axes = plt.subplots(2, n_snaps, figsize=(n_snaps * 2.8, 6))
    fig.patch.set_facecolor('#0d0d0d')
    fig.suptitle(f'Hopper  |  траектория #{idx}  |  split t={split_t}',
                 color='white', fontsize=12, y=1.01)

    row_labels = ['Ground Truth', 'Predicted']
    for row, frames in enumerate([frames_gt, frames_pred]):
        for col, t in enumerate(ts):
            ax = axes[row, col]
            ax.imshow(frames[t])
            ax.axis('off')
            phase = 'obs' if t < split_t else 'pred'
            color = '#7ab8e8' if t < split_t else '#e8a06a'
            ax.set_title(f't={t} [{phase}]', color=color, fontsize=8, pad=3)
        axes[row, 0].set_ylabel(row_labels[row], color='#cccccc',
                                fontsize=9, rotation=90, labelpad=30)

    plt.tight_layout()
    fname = f'hopper_snapshots_{idx}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    print(f'Сохранено: {fname}')
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    def find_project_root(marker_files=('run_models.py', 'mujoco_physics.py')):
        current = Path.cwd().resolve()
        for _ in range(6):
            if all((current / m).exists() for m in marker_files):
                return current
            for child in current.iterdir():
                if child.is_dir() and all((child / m).exists() for m in marker_files):
                    return child
            current = current.parent
        raise RuntimeError(f'Не могу найти корень проекта от {Path.cwd()}')

    ROOT = find_project_root()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    print('ROOT:', ROOT)

    # ── КОНФИГ ────────────────────────────────────────────────────────────────
    EXPERIMENT_ID = 25475   # None = последний checkpoint
    CKPT_DIR      = ROOT / 'experiments'
    # ─────────────────────────────────────────────────────────────────────────

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if EXPERIMENT_ID is None:
        ckpts = sorted(CKPT_DIR.glob('experiment_*.ckpt'), key=lambda p: p.stat().st_mtime)
        if not ckpts:
            raise FileNotFoundError(f'Нет checkpoint-ов в {CKPT_DIR}/')
        ckpt_path = ckpts[-1].resolve()
    else:
        ckpt_path = (CKPT_DIR / f'experiment_{EXPERIMENT_ID}.ckpt').resolve()

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str,
                        default=str(ROOT / 'data' / 'HopperPhysics' / 'training.pt'))
    parser.add_argument('--idx',  type=int, default=0,   help='Индекс траектории')
    parser.add_argument('--mode', type=str, default='compare',
                        choices=['compare', 'snapshots', 'sequential'],
                        help='compare=анимация side-by-side, snapshots=стоп-кадры, '
                             'sequential=10 траекторий подряд (5 train + 5 test)')
    parser.add_argument('--save', type=str, default=None,
                        choices=[None, 'gif', 'mp4'])
    parser.add_argument('--fps',  type=int, default=25)
    args = parser.parse_args()

    print(f'Загружаем данные: {args.train}')
    data, data_norm, att_min, att_max = load_data(args.train)
    print(f'Форма данных: {data.shape}')

    print(f'Загружаем модель: {ckpt_path}')
    model, ckpt_args = load_model(ckpt_path, ROOT, device)

    if args.mode == 'sequential':
        N = len(data)
        train_end = int(N * 0.8)
        train_indices = list(range(0,          min(5, train_end)))
        test_indices  = list(range(train_end,  min(train_end + 5, N)))
        print(f'Train indices: {train_indices}')
        print(f'Test  indices: {test_indices}')
        show_sequential(data, att_min, att_max, model, device, train_indices, test_indices,
                        fps=args.fps, save=args.save)
    else:
        traj_gt = data[args.idx]
        print(f'Траектория #{args.idx}: форма {traj_gt.shape}')
        print('Запускаем инференс...')
        pred_second_half, split_t = predict_second_half(model, traj_gt, att_min, att_max, device)
        print(f'split_t={split_t}, pred shape={pred_second_half.shape}')

        if args.mode == 'compare':
            show_gt_vs_pred(traj_gt, pred_second_half, split_t,
                            idx=args.idx, fps=args.fps, save=args.save)
        elif args.mode == 'snapshots':
            show_snapshots_gt_vs_pred(traj_gt, pred_second_half, split_t, idx=args.idx)
