# 実行コマンド

初回セットアップ、offline検証、単腕・双腕の起動・停止は[README.md](README.md)を参照してください。

双腕Runtime起動（ホスト側、既定はdry-run）：

```bash
bash scripts/run_dual.sh --checkpoint ~/piper_checkpoints/dnpdemo1_20261028
```

実行Actionを送信する場合は起動時に`--controller-run`を付け、READY後に制御コンテナ内で開始します。

```bash
rosservice call /piper_vla_dual_runtime/start \
  "prompt: 'Use the robot arms to pick up the silver pouch from the table, align the bundle neatly, and place it inside the cardboard box.'"
rosservice call /piper_vla_dual_runtime/stop '{}'
```

単腕は`run.sh`と`/piper_vla_runtime/start`・`/stop`を使用します。サーバー終了には起動時に生成されたComposeのパスを指定してください。
