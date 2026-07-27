# RF-DETR Training Dashboard

A zero-dependency, read-only dashboard for RF-DETR/Lightning runs.

```bash
python server.py \
  --training-root /root/autodl-tmp/FuckAnthropic/rfdetr_runs \
  --host 127.0.0.1 \
  --port 6006
```

It discovers the newest `metrics.csv`, `training_config.json`, and
`checkpoint*.pth` recursively under the training root. GPU telemetry comes from
`nvidia-smi`. The browser refreshes every three seconds. The latest metrics file
can be downloaded from the dashboard or directly from
`/api/download/metrics.csv`.

Competition submissions named `submission_epochNNN.csv` in the project root are
merged into the full Epoch results table. Rows with a generated submission are
highlighted and clicking anywhere on that row downloads its matching CSV. Epochs
without a generated submission remain visible with a `未生成` state. Downloads
use the allowlisted `/api/download/submission.csv?name=...` endpoint.

For AutoDL, connect from a local terminal with an SSH tunnel:

```bash
ssh -L 6006:127.0.0.1:6006 -p <SSH_PORT> root@<SSH_HOST>
```

Then open `http://127.0.0.1:6006`.

After an AutoDL instance restart, start the monitor again with:

```bash
cd /root/autodl-tmp/FuckAnthropic/training_dashboard
bash start.sh
```
