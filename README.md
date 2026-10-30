# PiPER VLA Runtime

学習済みCheckpointパッケージから、PiPERの単腕・双腕推論を実行するための共有フォルダです。モデルはフォルダ外に置き、Dockerコンテナから読み取り専用で参照します。

## 1. 環境の概要

| 項目 | 単腕 | 双腕 |
| --- | --- | --- |
| 起動スクリプト | `scripts/run.sh` | `scripts/run_dual.sh` |
| モデル | π0／π0.5、Full／LoRA | π0／π0.5、Full／LoRA |
| 状態・出力Action | 7次元（6関節＋gripper） | 14次元（左7＋右7） |
| カメラ | 正面・手首 | 正面・左手首・右手首 |
| Policyポート | `8000` | `8001` |
| Composeプロジェクト名 | `piper-vla` | `piper-vla-dual` |
| Policyコンテナ名 | `piper-vla-policy-1` | `piper-vla-dual-policy-1` |
| Runtimeコンテナ名 | `piper-vla-client-1` | `piper-vla-dual-client-1` |
| 実行Action送信先 | `/vla/approved_action` | `/vla_dual/approved_action` |
| HOLD送信先 | `/vla/hold` | `/vla_dual/hold` |

通常起動では次の2サービスを使用します。Dockerイメージは単腕・双腕で共通です。

- **`policy`**：GPU上でモデルを読み込み、WebSocketで推論要求に応答します。
- **`client`**：ROS1 NoeticのRuntimeです。観測をPolicyへ送り、結果を外部Controllerへ渡します。

ROS master、PiPER Controller、CAN制御、カメラ配信は別途用意してください。本フォルダには外部Controller・PiPER SDKは含まれません。

対応するのは関節Actionで、関節の単位はrad、gripperはmです。EE形式、未知のカメラ構成、混合正規化は対象外です。双腕ROS RuntimeはAction horizonが50、内部Action次元が32のモデルに限り、双腕π0.5は関節差分・絶対gripperの学習形式が必要です。8構成すべての実Checkpointでの動作を保証するものではないため、受領したモデルごとに後述の`--offline`で確認してください。

```text
piper_vla_runtime_share/
├── README.md
├── assets/                    共通tokenizer
├── configs/                   単腕・双腕の初期設定
├── docker/                    Dockerfile・Policyサーバー
├── scripts/                   ビルド・起動・検証
├── src/                       共通ローダー・単腕／双腕Runtime
├── ros/piper_vla_runtime_msgs/ ROSサービス型
├── vendor/openpi_pbl5/         推論に必要なOpenPI・client・依存lock
└── tests/                     回帰テスト
```

## 2. 初回だけ：環境構築

### 2.1 事前に用意するもの

- Ubuntu、NVIDIA GPUと対応ドライバ、モデルを読み込める空きVRAM
- Docker Engine、`gpus: all`に対応したDocker Compose、NVIDIA Container Toolkit
- `sudo`なしで`docker`を実行できる利用者設定
- Python 3.10以降、PyYAML
- 初回ビルド時のインターネット接続と、イメージ・Checkpoint用の空きディスク容量

以下は本フォルダをホーム直下の`~/piper_vla_runtime_share`へ配置した例です。別の場所に置いた場合は`cd`の行を変更してください。**ホスト**はUbuntu側のターミナル、**Docker内**は指定コンテナのシェルを意味します。`<checkpoint_name>`などの山括弧付き文字列は実際の名前に置き換えます。

**ホスト：**

```bash
cd "$HOME/piper_vla_runtime_share"
nvidia-smi
docker version
docker compose version

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install PyYAML
```

`venv`がない場合は、先に`sudo apt install python3-venv`で導入してください。ZIP形式を使う場合は`unzip`も必要です。

### 2.2 Dockerイメージのビルド

**ホスト・本フォルダ内：**

```bash
bash scripts/build.sh
```

既定のイメージ名は上記2つです。ビルド済みイメージを受領・ロード済みの場合は再ビルド不要です。名前を変更する場合は、ビルド時と起動時の両方で`PIPER_PACKAGE_IMAGE`と`PIPER_PACKAGE_CLIENT_IMAGE`を同じ値に設定します。

モデル変更のたびにビルドする必要はありません。通常起動は既存イメージを使用し、自動ビルド・pullを行いません。

## 3. Checkpointパッケージの配置

### 3.1 保存先と展開

保存先は`~/piper_checkpoints/<checkpoint_name>`を推奨します。`--checkpoint`には**`metadata.json`があるフォルダ**を指定します。起動時にそのフォルダ全体がコンテナの`/package`へ読み取り専用でマウントされるため、別の保存先も指定できます。

**ホスト：** まず受領したアーカイブの中身を確認します。使用する形式のコマンドだけ実行してください。

```bash
checkpoint_name='<checkpoint_name>'
checkpoint_dir="$HOME/piper_checkpoints/$checkpoint_name"
archive="$HOME/Downloads/${checkpoint_name}.tar.gz"
tar -tzf "$archive" | head -40
```

ZIPの場合：

```bash
archive="$HOME/Downloads/${checkpoint_name}.zip"
unzip -l "$archive" | head -40
```

アーカイブ直下に`metadata.json`と`checkpoint/`がある場合は、空の保存先を作って展開します。既存モデルを上書きしないよう、`mkdir`が成功した場合だけ展開します。

```bash
mkdir -p "$HOME/piper_checkpoints"
# tar.gzの場合
mkdir "$checkpoint_dir" && tar -xzf "$archive" -C "$checkpoint_dir"
```

ZIPの場合は上記の展開行の代わりに次を実行します。

```bash
mkdir "$checkpoint_dir" && unzip "$archive" -d "$checkpoint_dir"
```

アーカイブに親フォルダが含まれる場合は、その階層も展開されます。例えば`<checkpoint_name>/metadata.json`が入っていた場合、展開後は次のように指定先を1階層下へ変更してください。

```bash
checkpoint_dir="$HOME/piper_checkpoints/$checkpoint_name/$checkpoint_name"
```

### 3.2 正しいフォルダ構成

```text
<checkpoint_name>/             ← --checkpointに渡す場所
├── metadata.json
├── training.yaml
├── tuning_norm_config.json
├── norm_stats.json
├── MANIFEST.sha256.json
└── checkpoint/
    └── <step>/
        ├── _CHECKPOINT_METADATA
        ├── assets/<asset_id>/norm_stats.json
        ├── params/
        │   ├── _METADATA
        │   ├── _sharding
        │   ├── manifest.ocdbt
        │   ├── array_metadatas/...
        │   └── ...            Orbaxの重みデータ一式
        └── train_state/...    受領パッケージに含まれる場合は保持
```

`<asset_id>`には複数階層のパスを含む場合があります。複数stepがあるパッケージは`metadata.json`の`checkpoint_step`または`step`で選択されます。指定がない場合はstepフォルダが1つだけ必要です。起動引数にstep選択オプションはありません。

設定・統計・重みは受領時のまま保持してください。SHA256検証はファイル一覧も照合するため、`train_state`などを削除したり、ログやメモを追加したりすると検証に失敗します。ルートとCheckpoint内の`norm_stats.json`も一致が必要です。

LoRAもベース重みを含む完全なパラメータツリーが必要です。adapterだけのパッケージや旧Bundle形式は使用できません。ベースモデルの別取得・LoRAマージ・重み変換は不要です。

### 3.3 配置・整合性の確認（GPU不要）

**ホスト・本フォルダ内：** 前項から同じターミナルを使い、`checkpoint_dir`が正しいことを確認してください。

```bash
source .venv/bin/activate
ls "$checkpoint_dir"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" python3 - "$checkpoint_dir" <<'PY'
import sys
from piper_vla_common.checkpoint_package import inspect_package, runtime_manifest

spec = inspect_package(sys.argv[1])
print("パッケージ検証OK:", spec["robot"], spec["family"], spec["method"], "step", spec["step"])
print("使用Checkpoint:", spec["checkpoint"])
print("契約:", spec["contract"])
print("ROS Runtime:", runtime_manifest(spec).get("runtime_unsupported", "対応"))
PY
```

ファイル一覧・全ファイルのSHA256、設定、Orbaxの配列メタデータ、統計の整合性を確認します。モデルの読み込みやGPU推論は行いません。サイズによって時間がかかります。

### 3.4 モデルごとのGPU推論確認（実機に接続しない）

**ホスト・本フォルダ内：** モデルに合わせて片方だけ実行します。

```bash
# 単腕
bash scripts/run.sh --checkpoint "$checkpoint_dir" --offline
```

```bash
# 双腕
bash scripts/run_dual.sh --checkpoint "$checkpoint_dir" --offline
```

Policy用イメージだけで重みを復元し、模擬観測から3回推論します。ROS・カメラ・Controllerへ接続しません。成功時は`Offline report: /tmp/piper-package-.../gpu.json`が表示され、レポートに`"success": true`が記録されます。通常Runtimeとは別の一時コンテナでGPUを使用するため、既存サーバーと同時に実行できるか空きVRAMを確認してください。

## 4. 毎回：通常起動前の準備

**ホスト・本フォルダ内：** 新しいターミナルでは毎回実行してください。

```bash
cd "$HOME/piper_vla_runtime_share"
source .venv/bin/activate
checkpoint_name='<checkpoint_name>'
checkpoint_dir="$HOME/piper_checkpoints/$checkpoint_name"
```

親フォルダ付きで展開した場合は`checkpoint_dir`を実際の`metadata.json`のある場所に変更します。

ROS master、必要な観測配信、実機で使う外部Controllerを準備し、次の設定と合わせてください。dry-runでもROS masterと観測は必要です。ROSなしで確認する場合は`--offline`を使用します。

| 入力 | 単腕の既定トピック | 双腕の既定トピック |
| --- | --- | --- |
| 関節状態（JointState） | `/joint_states_single` | `/followerL/joint_states_single`、`/followerR/joint_states_single` |
| 正面画像（CompressedImage） | `/camera/front/compressed` | `/camera/front/compressed` |
| 手首画像（CompressedImage） | `/camera/wrist/compressed` | `/camera/left_wrist/compressed`、`/camera/right_wrist/compressed` |
| EE姿勢（PoseStamped） | `/end_pose`（関節モデルでも受信が必要） | 不要 |

双腕のJointStateの名前は`joint1`～`joint6`と`gripper`が必要です。双腕観測は既定で0.5秒以内、相互の時刻差0.15秒以内である必要があります。

トピックなどは起動前に`configs/runtime.yaml`（単腕）または`configs/runtime.dual.yaml`（双腕）で変更します。**dry-runは起動引数で上書きされる**ため、下記の起動コマンドで選択してください。

ROS masterとRuntimeが同じPCにある場合、既定値は`ROS_MASTER_URI=http://127.0.0.1:11311`、`ROS_IP=127.0.0.1`です。複数PCで使う場合は起動前に次を設定し、ROS各ノード間の相互通信・時刻同期も用意します。

```bash
export ROS_MASTER_URI='http://<ROSマスターのIP>:11311'
export ROS_IP='<この推論PCの到達可能なIP>'
```

同じ構成のサーバーが既に動いている場合は、第7節で対象を停止してから起動します。同じComposeプロジェクト名での起動は既存コンテナを更新する場合があります。

## 5. 単腕推論サーバーの起動

### 5.1 ホストで起動（片方を選択）
```bash
# 実機実行：Start後に外部ControllerへActionを送信する
bash scripts/run.sh --checkpoint "$checkpoint_dir" --controller-run
```

単腕モデルを要求し、Policyの起動・ウォームアップ後にRuntimeを起動します。外部Controller側も実機実行の設定が必要です。`--controller-run`は外部Controllerの設定を変更しません。

### 5.2 起動成功の確認（ホスト）

```bash
docker ps -a --filter 'label=com.docker.compose.project=piper-vla'
docker logs --tail 30 piper-vla-policy-1
docker logs --tail 30 piper-vla-client-1
docker exec piper-vla-client-1 bash -lc 'source /opt/ros/noetic/setup.bash && rosservice info /piper_vla_runtime/start'
```

起動スクリプトに`Policy WebSocket ready`と`READY (Start with prompt required); generated config: /tmp/piper-package-...`が表示され、両コンテナが`Up`、Startサービスが表示されることを確認します。**`generated config:`のパスは停止操作に使うため控えてください。** `READY`だけで実機の動作確認が済んだことにはなりません。

### 5.3 推論開始・停止（Docker内）

ホストからRuntimeコンテナのシェルを開きます。

```bash
docker exec -it piper-vla-client-1 bash
```

**以降はDocker内：** 指示文は学習した作業に合わせて変更します。実機実行モードではStartで動作が始まります。

```bash
source /opt/ros/noetic/setup.bash
rosservice call /piper_vla_runtime/start "prompt: 'Put all the green objects in the box.'"
```

作業を止めるときは、同じDocker内のシェルで実行します。

```bash
rosservice call /piper_vla_runtime/stop '{}'
exit
```

`success: true`は開始要求を受け付けたことを示します。観測不足・通信異常などはRuntimeログで確認します。`/stop`は指令を止めてHOLDを要求し、コンテナは起動したままです。`exit`は操作用シェルを閉じるだけです。

## 6. 双腕推論サーバーの起動

### 6.1 ホストで起動（片方を選択）

```bash
# 実機実行：Start後に外部ControllerへActionを送信する
bash scripts/run_dual.sh --checkpoint "$checkpoint_dir" --controller-run
```

双腕モデルを要求します。実機実行では、外部Controllerが`/vla_dual/controller_start`と`/vla_dual/controller_status`による開始応答・状態通知、および`/vla_dual/approved_action`と`/vla_dual/hold`に対応している必要があります。

### 6.2 起動成功の確認（ホスト）

```bash
docker ps -a --filter 'label=com.docker.compose.project=piper-vla-dual'
docker logs --tail 30 piper-vla-dual-policy-1
docker logs --tail 30 piper-vla-dual-client-1
docker exec piper-vla-dual-client-1 bash -lc 'source /opt/ros/noetic/setup.bash && rosservice info /piper_vla_dual_runtime/start'
```

単腕と同様に`Policy WebSocket ready`、`READY`、両コンテナの`Up`、Startサービスを確認し、`generated config:`のパスを控えます。

### 6.3 推論開始・停止（Docker内）

ホストからシェルを開きます。

```bash
docker exec -it piper-vla-dual-client-1 bash
```

**以降はDocker内：** 次の英語指示文は作業例です。実機実行モードではStartで両腕の動作が始まります。

```bash
source /opt/ros/noetic/setup.bash
rosservice call /piper_vla_dual_runtime/start \
  "prompt: 'Use the robot arms to pick up the silver pouch from the table, align the bundle neatly, and place it inside the cardboard box.'"
```

作業を止めるときは、同じDocker内のシェルで実行します。

```bash
rosservice call /piper_vla_dual_runtime/stop '{}'
exit
```

`/stop`は両腕へHOLDを要求し、コンテナは起動したままです。次の作業は新しい指示文で再度Startします。実行中に指示を変える場合も、先にStopしてください。

### 起動引数（単腕・双腕で共通）

| 引数 | 用途 |
| --- | --- |
| `--checkpoint PATH` | `metadata.json`があるパッケージルート。`--package PATH`も同じ意味 |
| `--dry-run` | 実行Actionを送信しない。省略時もこのモード |
| `--controller-run` | 外部Controllerへ実行Actionを送信する |
| `--offline` | ROSに接続せず、別コンテナでGPU推論を検証する |
| `--warmup-prompt TEXT` | 模擬観測による起動時の確認用指示文。実作業のStart指示とは別 |
| `--max-execution-steps N` | 双腕の送信Action数の上限。`0`は無制限。単腕では制限として機能しない |
| `--help` | 引数一覧を表示する |

`--checkpoint`省略時は、単腕が`~/piper_checkpoints/current`、双腕が`~/piper_checkpoints/current_dual`を参照します。本手順ではモデルの取り違えを避けるため毎回明示します。`--robot`と`--runtime-dir`はスクリプトが設定するため指定不要です。

## 7. 推論サーバー・コンテナの停止と再開

以下は通常起動した`policy`・`client`の操作です。**すべてホストで実行します。** 先に対象RuntimeのStopを呼び、その後にコンテナを停止します。

### 7.1 推論動作を止める（対象の片方だけ実行）

```bash
# 単腕
docker exec piper-vla-client-1 bash -lc 'source /opt/ros/noetic/setup.bash && rosservice call /piper_vla_runtime/stop "{}"'
```

```bash
# 双腕
docker exec piper-vla-dual-client-1 bash -lc 'source /opt/ros/noetic/setup.bash && rosservice call /piper_vla_dual_runtime/stop "{}"'
```

### 7.2 停止するComposeを選ぶ

**片方だけ**設定します。以降のコマンドは選んだプロジェクト内のサービスだけを対象にします。

```bash
# 単腕を操作する場合
runtime_project=piper-vla
policy_port=8000
```

```bash
# 双腕を操作する場合
runtime_project=piper-vla-dual
policy_port=8001
```

同じターミナルで、対象コンテナのラベルから生成Composeの場所を取得します。停止済みコンテナからも取得できます。

```bash
runtime_compose=$(docker inspect "${runtime_project}-policy-1" --format '{{ index .Config.Labels "com.docker.compose.project.config_files" }}')
printf '対象: %s\nCompose: %s\n' "$runtime_project" "$runtime_compose"
test -f "$runtime_compose"
docker compose --env-file /dev/null -p "$runtime_project" -f "$runtime_compose" ps -a
```

ラベルを取得できない場合は、起動時に控えたパスを使います。

```bash
runtime_compose='/tmp/piper-package-<起動時の文字列>/compose.yaml'
```

ファイルが存在しない場合はこのComposeを使った操作はできません。一時フォルダを削除する前に終了操作と必要なログの保存を済ませてください。

### 7.3 必要な停止操作を選ぶ

```bash
# Runtimeクライアントだけを停止（Policyは保持）
docker compose --env-file /dev/null -p "$runtime_project" -f "$runtime_compose" stop client
```

```bash
# Policyだけを停止（必ず先に7.1のStopを実行）
docker compose --env-file /dev/null -p "$runtime_project" -f "$runtime_compose" stop policy
```

```bash
# 両方を停止し、コンテナは残す
docker compose --env-file /dev/null -p "$runtime_project" -f "$runtime_compose" stop client policy
```

```bash
# 対象のPolicy・Runtimeを終了し、コンテナと専用ネットワークを削除
docker compose --env-file /dev/null -p "$runtime_project" -f "$runtime_compose" down
```

`down`は選んだ単腕または双腕の構成だけを対象にします。外部制御コンテナ、別プロジェクトのサーバー、モデル、Dockerイメージは削除しません。単腕・双腕の両方を終了する場合は、対象を選び直してそれぞれ実行します。

### 7.4 停止状態の確認

```bash
docker compose --env-file /dev/null -p "$runtime_project" -f "$runtime_compose" ps -a
docker ps -a --filter "label=com.docker.compose.project=$runtime_project"
```

`stop`後は対象サービスが`Exited`、`down`後は対象コンテナが表示されないことを確認します。端末を閉じたり、起動スクリプトをCtrl+Cしたりするだけでは、バックグラウンドのコンテナは終了しません。

### 7.5 同じCheckpointで再開

`stop`で停止した場合は、Policy→接続確認→Runtimeの順に再開します。第7.2節の変数を設定したターミナルで、本フォルダから実行してください。

```bash
docker compose --env-file /dev/null -p "$runtime_project" -f "$runtime_compose" start policy
python3 scripts/wait_policy_ready.py --host 127.0.0.1 --port "$policy_port"
```

`Policy WebSocket ready`を確認してから実行します。

```bash
docker compose --env-file /dev/null -p "$runtime_project" -f "$runtime_compose" start client
docker compose --env-file /dev/null -p "$runtime_project" -f "$runtime_compose" ps -a
```

停止していないサービスへの`start`は不要です。第5節または第6節のStartサービスで作業を開始します。`down`で削除した場合は、元の起動スクリプトを同じCheckpoint・同じ実行モードで実行します。

## 8. Checkpoint・単腕／双腕の切り替え

1. 第7.1節で現在のRuntimeの推論を停止します。
2. 第7.2節で現在の構成を選び、`down`でそのPolicy・Runtimeを終了します。
3. 新しいモデルを別名のフォルダへ配置し、第3.3節で検証します。
4. ホスト・本フォルダ内で、新しいパッケージの場所を設定します。

```bash
checkpoint_name='<新しいcheckpoint_name>'
checkpoint_dir="$HOME/piper_checkpoints/$checkpoint_name"
```

続いて、実行する構成・モードを1つ選びます。親フォルダ付きで展開した場合は、先に`checkpoint_dir`を調整してください。

```bash
# 単腕・dry-run
bash scripts/run.sh --checkpoint "$checkpoint_dir"
```

```bash
# 単腕・実機実行
bash scripts/run.sh --checkpoint "$checkpoint_dir" --controller-run
```

```bash
# 双腕・dry-run
bash scripts/run_dual.sh --checkpoint "$checkpoint_dir"
```

```bash
# 双腕・実機実行
bash scripts/run_dual.sh --checkpoint "$checkpoint_dir" --controller-run
```

Checkpoint切り替えではPolicyとRuntimeの両方を作り直し、モデル・統計・設定を揃えます。Dockerイメージの再ビルドは不要です。新しい`generated config:`のパスを控えてください。

単腕から双腕、双腕から単腕へ切り替える場合も、先に元の構成を終了し、新しい構成のモデル・カメラ・Controllerを準備して対応するスクリプトを使います。起動スクリプトはモデルの単腕／双腕の不一致をエラーにします。

## 9. ログとよくある確認事項

- 生成Compose・設定：`/tmp/piper-package-<起動ごとの文字列>/`。固定のComposeファイルは本フォルダ内にありません。
- 自動記録：生成フォルダの`logs/`。単腕はRuntimeと外部Controllerの両方が実機実行設定のとき、双腕は`recording.enabled`が有効なときに記録します。`LOG_ROOT`環境変数は現行の起動処理では使いません。
- 必要なログは`down`後もホスト側に残りますが、`/tmp`の清掃で失われるため別の保存先へコピーしてください。
- `metadata.json`が見つからない：アーカイブの親フォルダを含めて展開されていないか確認し、パッケージルートを指定します。
- SHA256・設定不一致：元のパッケージと展開結果を確認します。検証を通すためのファイル削除・統計編集は行いません。
- GPUメモリ不足：ホストの`nvidia-smi`で使用状況を確認します。別サーバーの停止はその管理者と調整してください。
- `READY`が出ない／Runtimeが終了する：対象の`docker logs`を確認します。起動失敗時もPolicyなどが残る場合があるため、第7節でその構成だけを終了します。
- 観測待ちになる：ROS masterへの接続、トピック名・型、配信の継続を確認します。単腕は`/end_pose`も必要です。

別の制御用ROS環境からStart／Stopする場合は、その環境に`ros/piper_vla_runtime_msgs/`を追加してcatkinでビルド・sourceしてください。本READMEの操作方法では、サービス型が導入済みのRuntimeコンテナを使うため追加ビルドは不要です。

OpenPIのライセンスは`vendor/openpi_pbl5/LICENSE`を参照してください。
