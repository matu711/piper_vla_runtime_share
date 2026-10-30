# PiPER VLA 学習・評価パイプライン

## 概要

本プロジェクトでは、robosuite上のPiPERロボットを用いて、赤色キューブを把持し、緑色キューブの上へ積み重ねるタスクを学習します。

VLAは次の対応関係を学習します。

```text
前方画像 + 手先画像 + 関節状態 + 言語指示
                    ↓
                  π0 VLA
                    ↓
今後のOSC行動
[dx, dy, dz, drx, dry, drz, gripper]
```

オブジェクト座標はデモ動作の生成には使用しますが、VLAへの直接入力には含めません。

---

## 全体の流れ

```text
【1. デモデータ収集】

MuJoCo / robosuite
├─ cubeA（赤色キューブ）の座標
├─ cubeB（緑色キューブ）の座標
├─ PiPER手先の位置・姿勢
└─ PiPERの現在状態
        ↓
ScriptedStackingPolicy
        ↓
OSC教師行動を生成
[dx, dy, dz, drx, dry, drz, gripper]
        ↓
成功したエピソードのみLeRobot形式で保存
├─ image
├─ wrist_image
├─ state
├─ actions
└─ task


【2. Fine-tuning】

LeRobotデータセット
        ↓
RepackTransform
├─ observation/image
├─ observation/wrist_image
├─ observation/state
├─ actions
└─ prompt
        ↓
前処理
├─ 状態・行動の正規化
├─ 画像を224×224へ変換
├─ promptをトークン化
└─ state/actionをモデル内部次元へパディング
        ↓
π0 VLA

入力
├─ 前方画像
├─ 手先画像
├─ state
└─ prompt

教師
└─ future actions: (10, 7)


【3. 評価】

robosuite
├─ 前方画像
├─ 手先画像
└─ state
        ↓
VLA推論サーバ
        ↓
今後10ステップ分の行動を予測
actions: (10, 7)
        ↓
先頭5ステップを実行
        ↓
最新画像・stateを再取得して再推論
        ↓
env.step(action)
        ↓
OSC_POSEコントローラ
        ↓
シミュレーション上のPiPERが動作
```

---

## 1. デモデータ収集

### 関連ファイル

```text
custom_scripts_piper/collect_piper_data.py
custom_scripts_piper/piper_training_params.yaml
custom_scripts_piper/piper_robot.py
```

### 実行方法

```bash
uv run custom_scripts_piper/collect_piper_data.py
```

### 教師行動の生成

`ScriptedStackingPolicy`は、MuJoCoから以下の正解情報を取得します。

| 情報          | 内容            |
| ----------- | ------------- |
| `cubeA_pos` | 赤色キューブの3次元座標  |
| `cubeB_pos` | 緑色キューブの3次元座標  |
| `eef_pos`   | PiPER手先の3次元位置 |
| `eef_quat`  | PiPER手先の姿勢    |
| `state`     | PiPERの関節状態    |

これらの情報を使って、接近、把持、持ち上げ、運搬、配置、解放の各動作を生成します。

```text
APPROACH
  ↓
ALIGN
  ↓
DESCEND
  ↓
GRASP
  ↓
LIFT
  ↓
TRANSPORT
  ↓
PLACE
  ↓
RELEASE
  ↓
DONE
```

`cubeA_pos`と`cubeB_pos`は教師行動の生成にのみ使用され、通常のVLA入力には保存されません。

### 保存されるデータ

各ステップで以下を保存します。

```python
frames.append({
    "image": front_img,
    "wrist_image": wrist_img,
    "state": state,
    "actions": action,
    "task": task_prompt,
})
```

| キー            |         Shape | 内容              |
| ------------- | ------------: | --------------- |
| `image`       | `(224,224,3)` | 前方カメラ画像         |
| `wrist_image` | `(224,224,3)` | 手先カメラ画像         |
| `state`       |        `(8,)` | 6関節角度＋グリッパ2関節位置 |
| `actions`     |        `(7,)` | OSC行動           |
| `task`        |           文字列 | 言語指示            |

### stateの内容

```text
state[0:6] = joint1〜joint6の角度
state[6]   = 左グリッパ関節位置
state[7]   = 右グリッパ関節位置
```

stateには以下は含まれません。

```text
cubeA座標
cubeB座標
手先位置・姿勢
関節速度・関節トルク
力覚・接触情報
現在のPhase
成功フラグ
```

### actionsの内容

```text
actions[0] = dx
actions[1] = dy
actions[2] = dz
actions[3] = drx
actions[4] = dry
actions[5] = drz
actions[6] = gripper
```

これは関節角度ではなく、`OSC_POSE`コントローラへ渡す手先差分指令です。

現在は姿勢制御を無効化しているため、基本的に次の3成分は0です。

```text
drx = 0
dry = 0
drz = 0
```

### 保存するエピソード

成功したエピソードだけを学習データへ保存します。

```yaml
data_collection:
  num_episodes: 300
```

この設定では、成功デモを300エピソード保存します。

### 保存先

```text
custom_scripts_piper/piper_tuning_data/local/piper_stack/
```

---

## 2. Fine-tuning

### 関連ファイル

```text
custom_scripts_piper/finetune_piper.py
src/openpi/training/config.py
scripts/compute_norm_stats.py
custom_scripts/train.py
```

### 実行方法

```bash
uv run custom_scripts_piper/finetune_piper.py
```

### 使用するモデル設定

```text
pi0_piper_stack_lora
```

定義場所：

```text
src/openpi/training/config.py
```

現在はPaliGemmaとAction Expertの両方にLoRAを使用します。

```python
paligemma_variant="gemma_2b_lora"
action_expert_variant="gemma_300m_lora"
```

### Fine-tuning時の入力

```text
observation/image
    前方カメラ画像

observation/wrist_image
    手先カメラ画像

observation/state
    joint1〜joint6とグリッパ2関節

prompt
    "stack the red cube on the green cube"
```

### 教師データ

```yaml
action_horizon: 10
```

現在時刻から未来10ステップ分の行動を教師として使用します。

```text
future actions: (10, 7)
```

学習する対応関係は次です。

```text
π0(image, wrist_image, state, prompt)
    → actions[t:t+10]
```

---

## 3. 評価

### 関連ファイル

```text
custom_scripts_piper/serve_piper_policy.py
custom_scripts_piper/eval_piper_vla_fixed.py
custom_scripts_piper/piper_training_params.yaml
```

### 推論サーバの起動

```bash
uv run custom_scripts_piper/serve_piper_policy.py
```

### 評価の実行

```bash
MUJOCO_GL=egl \
uv run custom_scripts_piper/eval_piper_vla_fixed.py \
  --config custom_scripts_piper/piper_training_params.yaml
```

### 評価時にVLAへ渡すもの

```python
{
    "observation/image": front_img,
    "observation/wrist_image": wrist_img,
    "observation/state": state,
    "prompt": task_prompt,
}
```

評価時にも、以下はVLAへ渡しません。

```text
cubeA_pos
cubeB_pos
eef_pos
eef_quat
Phase
reward
success
```

### VLAから返されるもの

```text
actions: (10, 7)
```

1ステップ分の行動は次です。

```text
[dx, dy, dz, drx, dry, drz, gripper]
```

### Action Chunk

```yaml
eval:
  action_chunk_size: 5
```

VLAが予測した10ステップのうち、最初の5ステップだけを実行します。

```text
10ステップ予測
        ↓
先頭5ステップ実行
        ↓
最新の画像・stateを取得
        ↓
再び10ステップ予測
```

途中で再観測するため、環境の変化に応じて行動を修正できます。

---

## 4. VLAからPiPERへの制御

評価コードでは、VLA出力をrobosuiteへ渡します。

```python
obs, reward, done, info = env.step(action)
```

その後の処理は次です。

```text
VLAの7次元OSC行動
        ↓
robosuite
        ↓
OSC_POSEコントローラ
        ↓
内部でPiPERの関節運動を計算
        ↓
シミュレーション上のPiPERが動作
```

VLAは関節角度を直接出力していません。

---

## 5. 評価結果

評価動画と結果は以下へ保存されます。

```text
custom_scripts_piper/piper_eval_videos/
├─ episode_0001.mp4
├─ episode_0002.mp4
├─ ...
└─ evaluation_summary.json
```

---

## まとめ

```text
デモ収集
画像・関節状態・言語指示・OSC行動を保存
        ↓
Fine-tuning
画像・state・promptから未来のOSC行動を学習
        ↓
評価
最新画像・state・promptをVLAへ入力
        ↓
VLAが10ステップ分のOSC行動を出力
        ↓
先頭5ステップを実行
        ↓
OSC_POSEでシミュレーションPiPERを制御
```

現在の構成では、オブジェクト座標をVLAへ直接入力せず、カメラ画像からオブジェクト位置を判断させます。
::: 
