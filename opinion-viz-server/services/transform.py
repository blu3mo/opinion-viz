# services/transform.py

import torch
import torch.nn as nn
import numpy as np

def fit_linear_transform(embeddings: np.ndarray, gpt4_dists: np.ndarray, n_components=5, 
                         lr=1e-2, max_epochs=1000, print_interval=100):
    """
    embeddings: shape (N, d)
    gpt4_dists: shape (N, N) (GPT-4のペアワイズ距離 0~1)
    n_components: 最終的に落とし込む次元(ここでは5)
    lr: 学習率
    max_epochs: 学習エポック数
    print_interval: 途中経過をprintする周期

    戻り値: (W, b) 
       - W: shape (d, n_components)
       - b: shape (n_components,)
    """
    device = torch.device("cpu")  # CPUで実行（GPU使うなら "cuda" に変更可）
    N, d = embeddings.shape

    # PyTorch のテンソル化
    X = torch.tensor(embeddings, dtype=torch.float32, device=device)  # (N, d)
    Dist = torch.tensor(gpt4_dists, dtype=torch.float32, device=device)  # (N, N)

    # 学習対象パラメータ: W, b
    # W: (d -> n_components), b: (n_components)
    W = nn.Parameter(torch.randn(d, n_components, device=device) * 0.01)
    b = nn.Parameter(torch.zeros(n_components, device=device))

    optimizer = torch.optim.Adam([W, b], lr=lr)

    for epoch in range(max_epochs):
        optimizer.zero_grad()

        # (N, n_components)
        Y = X @ W + b  # 線形変換: Y_i = W * X_i + b

        # Y_i - Y_j のユークリッド距離を計算
        # ただし直接二重ループすると遅いので工夫 or シンプルにループでもOK
        # ここは簡単にループで書きます
        loss = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist_ij = torch.dist(Y[i], Y[j])  # ユークリッド距離
                gpt_ij = Dist[i, j]
                loss += (dist_ij - gpt_ij)**2

        loss = loss / (N*(N-1)/2)  # 平均化

        loss.backward()
        optimizer.step()

        if (epoch+1) % print_interval == 0:
            print(f"Epoch {epoch+1}/{max_epochs}, Loss={loss.item():.4f}")

    # 学習後の W, b を numpy化して返す
    W_np = W.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    return (W_np, b_np)

def apply_linear_transform(embedding: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    embedding: (d,)
    W: (d, 5)
    b: (5,)
    => return: (5,)
    """
    return embedding @ W + b
