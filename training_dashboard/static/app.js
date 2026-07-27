const COLORS = {
  red: "#ff5e6c",
  blue: "#72a7ff",
  mint: "#45d7ac",
  orange: "#ff9b55",
  violet: "#b681ff",
  grid: "rgba(132,145,167,.16)",
  text: "#8491a7",
};

const $ = (id) => document.getElementById(id);
const fmt = (value, digits = 4) =>
  value === undefined || value === null || Number.isNaN(Number(value))
    ? "—"
    : Number(value).toFixed(digits);
const compact = (value) => {
  if (value === undefined || value === null) return "—";
  return Number(value).toLocaleString("zh-CN", { maximumFractionDigits: 1 });
};

function statusLabel(status) {
  return { training: "训练中", completed: "已完成", waiting: "等待训练" }[status] || "未知";
}

function setupCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width: rect.width, height: rect.height };
}

function drawChart(canvas, history, series, options = {}) {
  const { context: ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 45, right: 16, top: 12, bottom: 28 };
  const chartW = Math.max(1, width - pad.left - pad.right);
  const chartH = Math.max(1, height - pad.top - pad.bottom);
  const values = history.flatMap((row) =>
    series.map((item) => row[item.key]).filter((value) => value !== undefined && value !== null)
  );

  if (!values.length) {
    ctx.fillStyle = COLORS.text;
    ctx.font = "12px system-ui";
    ctx.textAlign = "center";
    ctx.fillText("等待训练指标", width / 2, height / 2);
    return;
  }

  let min = options.zeroBased ? 0 : Math.min(...values);
  let max = Math.max(...values);
  if (options.fixedMax) max = options.fixedMax;
  if (options.f1Scale) {
    min = Math.max(0, Math.floor((min - 0.03) * 20) / 20);
    max = Math.min(1, Math.ceil((max + 0.03) * 20) / 20);
  }
  if (max === min) { max += 0.05; min -= 0.05; }

  ctx.strokeStyle = COLORS.grid;
  ctx.fillStyle = COLORS.text;
  ctx.lineWidth = 1;
  ctx.font = "10px system-ui";
  ctx.textAlign = "right";
  for (let index = 0; index <= 4; index += 1) {
    const y = pad.top + (chartH * index) / 4;
    const value = max - ((max - min) * index) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillText(value.toFixed(value >= 10 ? 1 : 3), pad.left - 8, y + 3);
  }

  const maxEpoch = Math.max(1, ...history.map((row) => row.epoch || 0));
  series.forEach((item) => {
    const points = history
      .filter((row) => row[item.key] !== undefined && row[item.key] !== null)
      .map((row) => ({
        x: pad.left + (((row.epoch || 1) - 1) / Math.max(1, maxEpoch - 1)) * chartW,
        y: pad.top + ((max - row[item.key]) / (max - min)) * chartH,
      }));
    if (!points.length) return;
    ctx.beginPath();
    points.forEach((point, index) =>
      index === 0 ? ctx.moveTo(point.x, point.y) : ctx.lineTo(point.x, point.y)
    );
    ctx.strokeStyle = item.color;
    ctx.lineWidth = item.width || 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.stroke();
    const last = points[points.length - 1];
    ctx.fillStyle = item.color;
    ctx.beginPath();
    ctx.arc(last.x, last.y, 3, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.fillStyle = COLORS.text;
  ctx.textAlign = "center";
  ctx.fillText("Epoch 1", pad.left, height - 7);
  ctx.fillText(`Epoch ${maxEpoch}`, width - pad.right, height - 7);
}

function renderMilestones(epoch, total) {
  const marks = [0.25, 0.5, 0.75, 1];
  $("milestones").innerHTML = marks
    .map((fraction) => {
      const value = Math.max(1, Math.round(total * fraction));
      return `<div class="milestone ${epoch >= value ? "done" : ""}"><span>Epoch ${value}</span></div>`;
    })
    .join("");
}

function renderTable(history, submissions, watcher = {}) {
  const rowsByEpoch = new Map(
    history.map((row) => [Number(row.epoch), { ...row }])
  );
  submissions.forEach((submission) => {
    const epoch = Number(submission.epoch);
    const row = rowsByEpoch.get(epoch) || { epoch };
    row.submission = submission;
    rowsByEpoch.set(epoch, row);
  });
  const rows = [...rowsByEpoch.values()].sort((left, right) => right.epoch - left.epoch);
  $("rowCount").textContent = `${rows.length} epochs · ${submissions.length} CSV 可下载`;
  $("metricsRows").innerHTML = rows.length
    ? rows
        .map(
          (row) => {
            const submission = row.submission;
            const startedFrom = Number(watcher.started_from_epoch);
            const isHistorical =
              Number.isFinite(startedFrom) && row.epoch < startedFrom;
            const isGenerating =
              watcher.status === "generating" &&
              Number(watcher.current_epoch) === row.epoch;
            const attributes = submission
              ? `class="downloadable-row" tabindex="0" role="link" data-download-url="${submission.download_url}" data-download-name="${submission.name}" aria-label="下载 Epoch ${row.epoch} 提交 CSV"`
              : 'class="pending-row"';
            const action = submission
              ? `<span class="download-pill">下载 ${submission.name}</span>`
              : `<span class="pending-pill">${
                  isHistorical
                    ? "历史权重未保存"
                    : isGenerating
                      ? "生成中"
                      : "等待自动生成"
                }</span>`;
            return `<tr ${attributes}>
            <td>${row.epoch}</td>
            <td>${fmt(row.precision)}</td>
            <td>${fmt(row.recall)}</td>
            <td>${fmt(row.f1)}</td>
            <td>${fmt(row.map50)}</td>
            <td>${fmt(row.map5095)}</td>
            <td>${fmt(row.loss)}</td>
            <td>${action}</td>
          </tr>`;
          }
        )
        .join("")
    : '<tr class="empty-row"><td colspan="8">等待首轮验证结果</td></tr>';
}

function renderMetricsDownload(metricsDownloadUrl) {
  const metricsLink = $("downloadMetrics");
  if (metricsDownloadUrl) {
    metricsLink.href = metricsDownloadUrl;
    metricsLink.setAttribute("download", "rfdetr_metrics_current.csv");
    metricsLink.setAttribute("aria-disabled", "false");
    metricsLink.classList.remove("disabled");
  } else {
    metricsLink.removeAttribute("href");
    metricsLink.removeAttribute("download");
    metricsLink.setAttribute("aria-disabled", "true");
    metricsLink.classList.add("disabled");
  }
}

function renderConfig(parameters) {
  const values = [
    `${parameters.resolution || 1360}px`,
    parameters.precision || "BF16",
    `effective batch ${parameters.effective_batch || 16}`,
    `${parameters.scheduler || "cosine"} LR`,
    parameters.ema ? "EMA" : "No EMA",
    parameters.holdout || "25% holdout",
    parameters.tiling || "tiled inference",
  ];
  $("configChips").innerHTML = values.map((value) => `<span>${value}</span>`).join("");
}

function renderCheckpoints(checkpoints) {
  $("checkpoints").innerHTML = checkpoints.length
    ? checkpoints
        .slice(0, 5)
        .map((item) => `<span>${item.name} · ${item.size_mb} MB</span>`)
        .join("")
    : "<span>尚未生成</span>";
}

function render(data) {
  latestData = data;
  const current = data.current || {};
  const best = data.best || {};
  const gpu = data.gpu || {};
  const epoch = data.epoch || 0;
  const total = data.total_epochs || 100;
  const percent = Math.min(100, (epoch / total) * 100);

  $("modelName").textContent = data.model || "RF-DETR-2XL";
  $("statusText").textContent = statusLabel(data.status);
  $("statusDot").className = `status-dot ${data.status || ""}`;
  $("processText").textContent = data.process?.pid
    ? `PID ${data.process.pid} · 已运行 ${Math.floor(data.process.elapsed_seconds / 60)} 分钟`
    : data.metrics_file
      ? "日志已发现，等待下一次更新"
      : "等待 RF-DETR 训练启动";
  $("subline").textContent = `每 3 秒自动刷新 · ${gpu.name || "GPU 状态读取中"}`;
  $("epoch").textContent = epoch || "—";
  $("totalEpochs").textContent = total;
  $("epochBest").textContent = best.epoch ? `最佳 Epoch ${best.epoch}` : "尚无最佳轮次";
  $("currentF1").textContent = fmt(current.f1);
  $("bestF1").textContent = best.f1 !== undefined ? `最佳 ${fmt(best.f1)}` : "最佳 —";
  $("map5095").textContent = fmt(current.map5095);
  $("map50").textContent = `mAP50 ${fmt(current.map50)}`;
  $("precision").textContent = fmt(current.precision);
  $("recall").textContent = fmt(current.recall);
  $("gpuUsage").textContent = gpu.utilization === undefined ? "—" : `${compact(gpu.utilization)}%`;
  $("gpuMemory").textContent =
    gpu.memory_used_mib === undefined
      ? "显存 —"
      : `显存 ${(gpu.memory_used_mib / 1024).toFixed(1)} / ${(gpu.memory_total_mib / 1024).toFixed(0)} GB`;
  $("gpuTemp").textContent = gpu.temperature === undefined ? "温度 —" : `${compact(gpu.temperature)}°C`;
  $("progressBar").style.width = `${percent}%`;
  $("progressLabel").textContent = epoch ? `${epoch} / ${total} epochs (${percent.toFixed(1)}%)` : "等待训练日志";
  $("lastUpdate").textContent = `更新于 ${new Date(data.timestamp * 1000).toLocaleTimeString("zh-CN")}`;

  renderMilestones(epoch, total);
  renderTable(
    data.history || [],
    data.submissions || [],
    data.submission_watcher || {}
  );
  renderMetricsDownload(data.metrics_download_url);
  renderConfig(data.parameters || {});
  renderCheckpoints(data.checkpoints || []);
  drawChart($("f1Chart"), data.history || [], [
    { key: "f1", color: COLORS.red, width: 2.5 },
    { key: "precision", color: COLORS.blue },
    { key: "recall", color: COLORS.mint },
  ], { f1Scale: true });
  drawChart($("mapChart"), data.history || [], [
    { key: "map50", color: COLORS.blue },
    { key: "map5095", color: COLORS.orange },
  ], { f1Scale: true });
  drawChart($("lossChart"), data.history || [], [
    { key: "loss", color: COLORS.mint },
    { key: "loss_bbox", color: COLORS.violet },
  ]);
}

async function refresh() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (error) {
    $("statusText").textContent = "连接中断";
    $("processText").textContent = "正在重试监控服务";
    $("statusDot").className = "status-dot";
  }
}

let latestData = null;
$("metricsRows").addEventListener("click", (event) => {
  const row = event.target.closest(".downloadable-row");
  if (!row) return;
  const link = document.createElement("a");
  link.href = row.dataset.downloadUrl;
  link.download = row.dataset.downloadName;
  document.body.append(link);
  link.click();
  link.remove();
});
$("metricsRows").addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const row = event.target.closest(".downloadable-row");
  if (!row) return;
  event.preventDefault();
  row.click();
});
window.addEventListener("resize", () => latestData && render(latestData));
refresh();
setInterval(refresh, 3000);
