# 機械学習モデルの理論実装とライブラリ実装の比較

本リポジトリは、代表的な機械学習手法について以下の 3 つのアプローチを比較・実装した学習用ポートフォリオです：

- **理論に基づいた手動実装（from scratch）**
- **ライブラリ（scikit-learn / PyTorch）による実装**
- **数式・ロジックの可視化と比較分析**

---

## 🔧 対象アルゴリズム一覧（`notebooks/`）

| 分野             | モデル                            | 該当ファイル                                            |
| ---------------- | --------------------------------- | ------------------------------------------------------- |
| 回帰             | 線形回帰 / ロジスティック回帰     | `linear_regression.ipynb` / `logistic_regression.ipynb` |
| 分類             | 決定木                            | `decision_tree.ipynb`                                   |
| クラスタリング   | K-means                           | `k-means.ipynb`                                         |
| 異常検知         | 多変量ガウス分布                  | `anomaly_detection.ipynb`                               |
| ニューラルネット | 単層 / 多層ニューラルネットワーク | `neural_network_regularization.ipynb`                   |

## 📁 ディレクトリ構成

```
machine_learning/
│
├── notebooks/ # 各モデルの実験・可視化用ノートブック
│ ├── linear_regression.ipynb
│ ├── logistic_regression.ipynb
│ ├── decision_tree.ipynb
│ ├── k-means.ipynb
│ ├── anomaly_detection.ipynb
│ └── neural_network_regularization.ipynb
│
├── datasets/ # 使用した CSV 等の小規模データセット
│ └── diabetes.csv
│
├── README.md # ← 今このファイル
└── requirements.txt # 必要なライブラリ
```

---

## 🧠 モデルごとの特徴・評価指標

各ノートブックには以下の項目を記載しています：

- モデルの理論背景と数式
- 実装ロジックの詳細（手動 vs ライブラリ）
- 可視化による直感的理解
- 評価指標（MSE, Accuracy, F1 スコア など）
- 実験結果と考察

---

## 💡 工夫ポイント

- **手動実装**：理論に忠実なベクトル計算を行い、挙動の詳細を可視化
- **ライブラリとの比較**：出力結果や精度を比較し、それぞれのメリットを明確化
- **ニューラルネット**：PyTorch と scikit-learn を併用し、学習率・正則化の影響を実験
- **異常検知**：多変量ガウス分布に基づくスコアリング + 最適な閾値探索（F1 スコア）

---

## 💻 実行方法

```bash
# 必要ライブラリのインストール
pip install -r requirements.txt

# Jupyter Lab / Notebook を起動
jupyter lab
```
