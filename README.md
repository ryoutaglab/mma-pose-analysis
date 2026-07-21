# 格闘技姿勢解析ツール

日本拳法・BJJの経験者が、競技の本質から設計した格闘技特化の姿勢解析ツールです。  
YOLO26によるリアルタイムポーズ推定を活用し、試合・練習映像から選手の姿勢・重心軸・関節角度を可視化します。

---

## 特徴

- **格闘技特化の設計**  
  打撃・組技・寝技それぞれの局面を想定し、構えの崩れ・重心移動・関節の角度変化を読み取ることを目的として設計しています。

- **正中線（体軸）の可視化**  
  肩中点と股関節中点を結ぶ正中線をリアルタイムで描画し、体軸のブレや傾きを直感的に確認できます。

- **ByteTrackによるP1・P2の追跡**  
  ByteTrackでID追跡を行い、試合開始時の左右位置からP1・P2を自動割当。組み合い等で一時的にIDを見失っても、直前の座標との距離をもとに同一人物として復帰します。手動でP1・P2を指定するモードも用意しています。

- **動画ファイルへの書き出し**  
  解析結果をMP4ファイルとして保存します。ファイル名には解析開始日時（yyyymmddHHMMSS）が自動付与されます。

---

## 必要環境

| 項目 | バージョン |
|------|-----------|
| Python | 3.9 以上 |
| OpenCV | 4.x 以上 |
| Ultralytics (YOLO26) | 8.4 以上 |
| NumPy | 1.x 以上 |
| CUDA（任意） | GPU使用時 |

---

## インストール方法

```bash
# リポジトリをクローン
git clone https://github.com/ryoutaglab/mma-pose-analysis.git
cd mma-pose-analysis

# 依存パッケージをインストール
pip install ultralytics opencv-python numpy
```

YOLO26のモデルファイル（`yolo26x-pose.pt`）は初回実行時に自動でダウンロードされます。

---

## ファイル構成

```
mma-pose-analysis/
├── pose_analysis_mma.py   # 格闘技特化版（メイン）
├── sample.mp4             # サンプル動画（別途用意）
├── README.md
└── LICENSE
```

---

## 使い方

### 引数一覧を確認する

```bash
python pose_analysis_mma.py -h
```

### 動画ファイルを解析する（自動モード）

試合開始時に2人以上検出された時点で、画面左右の位置からP1・P2を自動割当します。

```bash
python pose_analysis_mma.py sample.mp4
```

引数を省略した場合は `sample.mp4` が自動で選択されます。

```bash
python pose_analysis_mma.py
```

### P1・P2を手動で指定する

`--track`（または `-T`）を付けると、最初のフレームを静止表示し、検出されたIDをクリックしてP1・P2を指定できます。

```bash
python pose_analysis_mma.py sample.mp4 --track
```

### ID消失後の補完時間を変更する

組み合い等でIDを一時的に見失った場合に、同一人物として復帰を試みる時間（秒）を指定できます（デフォルト3秒）。

```bash
python pose_analysis_mma.py sample.mp4 --track-buffer 5
```

### Webカメラでリアルタイム解析する

```bash
python pose_analysis_mma.py 0
```

### 出力ファイル

解析結果は自動的に動画ファイルとして保存されます。

```
sample_20260626143052.mp4
```

---

## 操作方法

### 解析中

| キー | 動作 |
|------|------|
| `q`  | 解析を終了する |

### P1・P2手動指定画面（`--track`使用時）

| キー | 動作 |
|------|------|
| クリック | 1人目＝P1、2人目＝P2として選択 |
| `Enter`  | 選択を確定する（2人選択後） |
| `Esc`    | クリックをやり直す |
| `q`      | 手動指定をキャンセルして終了する |

---

## 解析内容

| 描画要素 | 説明 |
|----------|------|
| キーポイント（点） | 17箇所の関節位置 |
| スケルトン（線） | 関節間の接続線 |
| 正中線（黄色太線） | 肩中点 → 股関節中点、および肩中点 → 両耳中点を結ぶ体軸 |
| 軸点（赤丸） | 正中線の両端点 |

P1（黄）・P2（オレンジ）をByteTrackのIDで追跡し、それぞれ異なる色で表示します。

---

## ライセンス

[GNU Affero General Public License v3.0](LICENSE)

本ツールは [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics)（AGPL-3.0）に依存しているため、本リポジトリ全体もAGPL-3.0で公開しています。改変・再配布、およびネットワーク経由での提供（Webサービス化等）を行う場合は、AGPL-3.0の条件に従いソースコードを公開する必要があります。

---

## 作者

**Ryota Goto (RyouTag Lab)**  
日本拳法・BJJ経験者
