# Mahjong RL Agent — Hướng dẫn sử dụng

## 1. Cài đặt

```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Cài đặt dependencies
pip install "stable-baselines3[extra]>=2.0" \
            "sb3-contrib>=2.0" \
            "gymnasium>=0.26" \
            tensorboard \
            numpy
```

## 2. Cấu trúc thư mục

```
.
├── mahjong_env.py      ← Gymnasium environment + shanten calculator
├── train_mahjong.py    ← Script huấn luyện & đánh giá
├── logs/               ← TensorBoard logs (tự tạo)
├── checkpoints/        ← Model checkpoint mỗi 100k steps (tự tạo)
└── models/             ← Model cuối cùng (tự tạo)
```

## 3. Chạy huấn luyện

```bash
# Mặc định: 1,000,000 steps, 8 envs song song
python train_mahjong.py --mode train

# Tùy chỉnh số steps và số envs
python train_mahjong.py --mode train --timesteps 2000000 --n-envs 16

# Dùng SubprocVecEnv (đa tiến trình, nhanh hơn trên CPU nhiều nhân)
python train_mahjong.py --mode train --subproc
```

## 4. Xem TensorBoard

```bash
tensorboard --logdir ./logs
# Mở trình duyệt: http://localhost:6006
```

**Các metric quan trọng cần theo dõi:**
| Metric | Ý nghĩa |
|--------|---------|
| `rollout/ep_rew_mean` | Reward trung bình mỗi episode |
| `rollout/ep_len_mean` | Số bước trung bình mỗi episode |
| `train/entropy_loss`  | Entropy của policy (càng cao = explore nhiều hơn) |
| `train/clip_fraction` | Tỉ lệ clipping PPO (lý tưởng ~0.1–0.2) |
| `mahjong/mean_shanten`| Shanten trung bình (càng thấp càng tốt, mục tiêu → 0) |

## 5. Chạy đánh giá

```bash
# Đánh giá model tốt nhất, 5 ván, in chi tiết từng bước
python train_mahjong.py --mode eval --model ./models/best/best_model

# Tắt in chi tiết từng bước (chỉ xem tổng kết)
python train_mahjong.py --mode eval --no-render --n-episodes 20
```

## 6. Train rồi eval liên tiếp

```bash
python train_mahjong.py --mode both --timesteps 1000000 --n-envs 8
```

## 7. Hyperparameter tuning gợi ý

| Hyperparameter | Mặc định | Gợi ý thử |
|---------------|----------|-----------|
| `learning_rate` | 3e-4 | 1e-4 — 5e-4 |
| `n_steps` | 2048 | 1024, 4096 |
| `batch_size` | 256 | 128, 512 |
| `ent_coef` | 0.01 | 0.005 — 0.05 |
| `net_arch` | [256,256] | [128,128], [512,256] |
| Reward win | +10 | +20, +50 |
| Reward step | -0.1 | -0.01, -0.05 |
| Shanten bonus | +0.5/step | +1.0, +2.0 |

## 8. Lưu ý kỹ thuật

- **ActionMasker** bọc env và cung cấp mask cho MaskablePPO qua `mask_fn`.
- **EvalCallback** tự động lưu model tốt nhất vào `./models/best/`.
- **CheckpointCallback** lưu model mỗi 100k steps để có thể resume.
- Để resume training từ checkpoint:
  ```python
  model = MaskablePPO.load("./checkpoints/mahjong_ppo_500000_steps", env=vec_env)
  model.learn(total_timesteps=500_000, reset_num_timesteps=False)
  ```
