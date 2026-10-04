/* 施肥处方图生成系统 - 前端逻辑 */
"use strict";

const state = {
  config: null,
  session: null,
  result: null,
  validated: false,
  running: false,
  currentView: "input",
  pollTimer: null,
  stageId: "1",
  annotated: {},
};

const $ = (id) => document.getElementById(id);
const fmtSize = (bytes) => {
  if (bytes == null) return "-";
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + " MB";
  return (bytes / 1073741824).toFixed(2) + " GB";
};

function toast(message, isError) {
  const el = $("toast");
  el.textContent = message;
  el.classList.toggle("error", !!isError);
  el.classList.remove("hidden");
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.classList.add("hidden"), isError ? 5000 : 2600);
}

async function api(url, body, method) {
  const opts = body
    ? { method: method || "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : { method: method || "GET" };
  const resp = await fetch(url, opts);
  let data = null;
  try { data = await resp.json(); } catch (e) { /* non-json */ }
  if (!resp.ok) throw new Error((data && data.error) || `请求失败 (${resp.status})`);
  return data;
}

/* ---------------- 文件名识别（与服务端一致，仅用于即时反馈） ---------------- */
function recognizeRole(name) {
  if (!/\.(tif|tiff)$/i.test(name)) return null;
  const stem = name.replace(/\.(tif|tiff)$/i, "").toLowerCase();
  if (stem.includes("rededge")) return "B1";
  if (stem.includes("green")) return "B2";
  if (stem.includes("nir")) return "B4";
  if (stem.includes("red")) return "B3";
  return "preview";
}

const ROLE_LABELS = {
  B1: "B1 · 红边 RedEdge",
  B2: "B2 · 绿光 Green",
  B3: "B3 · 红光 Red",
  B4: "B4 · 近红外 NIR",
  preview: "整体预览图（不参与诊断）",
};

/* ---------------- 数据接入 ---------------- */
function initInput() {
  const dropzone = $("dropzone");
  $("btn-pick-files").addEventListener("click", (e) => { e.stopPropagation(); $("file-input").click(); });
  $("btn-pick-folder").addEventListener("click", (e) => { e.stopPropagation(); $("folder-input").click(); });
  dropzone.addEventListener("click", () => $("file-input").click());
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    handleFileSelection(Array.from(e.dataTransfer.files));
  });
  $("file-input").addEventListener("change", (e) => {
    handleFileSelection(Array.from(e.target.files));
    e.target.value = "";
  });
  $("folder-input").addEventListener("change", (e) => {
    handleFileSelection(Array.from(e.target.files));
    e.target.value = "";
  });

  $("input-tabs").addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (!btn) return;
    document.querySelectorAll("#input-tabs .tab").forEach((t) => t.classList.toggle("active", t === btn));
    $("pane-upload").classList.toggle("hidden", btn.dataset.tab !== "upload");
    $("pane-folder").classList.toggle("hidden", btn.dataset.tab !== "folder");
  });

  $("btn-scan-folder").addEventListener("click", async () => {
    const path = $("folder-path").value.trim();
    if (!path) { toast("请填写文件夹路径", true); return; }
    try {
      const data = await api("/api/local/folder", { path });
      state.session = data;
      renderFiles();
      resetRoi();
      toast(`扫描完成：${Object.keys(data.files).length} 个文件已识别`);
    } catch (err) { toast(err.message, true); }
  });

  $("btn-validate").addEventListener("click", doValidate);
  $("btn-reset").addEventListener("click", async () => {
    try {
      const data = await api("/api/reset", {});
      state.session = data.session;
      state.result = null;
      state.validated = false;
      renderFiles();
      hideResultCards();
      resetViewer();
      resetRoi();
      toast("会话已清空");
    } catch (err) { toast(err.message, true); }
  });
}

function handleFileSelection(fileList) {
  const tifs = fileList.filter((f) => /\.(tif|tiff)$/i.test(f.name));
  const skipped = fileList.length - tifs.length;
  if (skipped > 0) toast(`已忽略 ${skipped} 个非 tif 文件`);
  if (!tifs.length) return;
  uploadQueue(tifs);
}

async function uploadQueue(files) {
  const queue = $("upload-queue");
  const items = files.map((f) => {
    const div = document.createElement("div");
    div.className = "upload-item";
    div.innerHTML = `
      <div class="u-row"><span>${f.name}</span><span class="u-status">排队中</span></div>
      <div class="u-bar"><div class="u-fill"></div></div>`;
    queue.appendChild(div);
    return { file: f, el: div };
  });

  for (const item of items) {
    await uploadOne(item);
  }
  await refreshSession();
}

async function uploadOne({ file, el }) {
  const fill = el.querySelector(".u-fill");
  const status = el.querySelector(".u-status");
  try {
    status.textContent = "开始上传";
    const start = await api("/api/upload/start", { name: file.name, size: file.size });
    const uploadId = start.upload_id;
    const chunkSize = 16 * 1024 * 1024;
    let offset = 0;
    while (offset < file.size) {
      const blob = file.slice(offset, offset + chunkSize);
      const resp = await fetch(`/api/upload/chunk/${uploadId}?offset=${offset}`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: blob,
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.error || "分块上传失败");
      offset = data.received;
      const pct = Math.round((offset / file.size) * 100);
      fill.style.width = pct + "%";
      status.textContent = `上传中 ${pct}%`;
    }
    status.textContent = "识别通道...";
    const fin = await api("/api/upload/finish", { upload_id: uploadId });
    state.session = fin.session;
    resetRoi();
    fill.style.width = "100%";
    status.textContent = `✓ ${fin.role_label || fin.role}`;
    el.classList.add("done");
  } catch (err) {
    status.textContent = "✗ " + err.message;
    el.classList.add("failed");
  }
}

async function refreshSession() {
  try {
    state.session = await api("/api/session");
    renderFiles();
    updateRoiControls();
    renderViewerStats();
  } catch (err) { /* ignore */ }
}

function renderFiles() {
  const container = $("file-table");
  container.innerHTML = "";
  const s = state.session || { files: {}, conflicts: [], missing: [] };
  const roles = ["preview", "B1", "B2", "B3", "B4"];

  for (const role of roles) {
    const info = s.files[role];
    const row = document.createElement("div");
    if (info) {
      row.className = "file-row" + (role === "preview" ? " preview" : "");
      row.innerHTML = `
        <span class="role-badge">${info.role_label || ROLE_LABELS[role]}</span>
        <span class="f-name" title="${info.name}">${info.name}</span>
        <span class="f-size">${fmtSize(info.size)}</span>
        <span class="f-src">${info.source === "upload" ? "已上传" : "本地引用"}</span>
        <button class="f-remove" title="移除">×</button>`;
      row.querySelector(".f-remove").addEventListener("click", async () => {
        try {
          state.session = await api("/api/files/remove", { role });
          renderFiles();
          resetRoi();
        } catch (err) { toast(err.message, true); }
      });
    } else if (role !== "preview") {
      row.className = "file-row missing";
      row.innerHTML = `
        <span class="role-badge">${ROLE_LABELS[role]}</span>
        <span class="f-name">缺少文件（文件名需含 ${role === "B1" ? "RedEdge" : role === "B2" ? "Green" : role === "B3" ? "Red" : "NIR"}）</span>`;
    } else {
      row.className = "file-row missing";
      row.innerHTML = `
        <span class="role-badge">${ROLE_LABELS[role]}</span>
        <span class="f-name">未上传（可选，缺失时用波段合成预览）</span>`;
    }
    container.appendChild(row);
  }

  for (const conflict of s.conflicts || []) {
    const row = document.createElement("div");
    row.className = "file-row conflict-note";
    row.innerHTML = `<span class="role-badge">${conflict.role_label}</span>
      <span class="f-name">重复文件已忽略：${conflict.name}</span>`;
    container.appendChild(row);
  }

  updateRunButton();
}

/* ---------------- 校验 ---------------- */
async function doValidate() {
  try {
    const data = await api("/api/validate", {});
    state.session = data.session;
    renderFiles();
    renderValidation(data);
    if (!data.errors.length) {
      $("card-params").classList.remove("hidden");
      toast("校验通过，正在生成整体预览图...");
      pollJob();
    } else {
      toast("校验未通过，请检查输入", true);
    }
  } catch (err) { toast(err.message, true); }
}

function renderValidation(data) {
  $("card-validation").classList.remove("hidden");
  const body = $("validation-body");
  body.innerHTML = "";

  if (data.errors.length) {
    const div = document.createElement("div");
    div.className = "alert error";
    div.innerHTML = "<b>校验未通过：</b><br>" + data.errors.map(escapeHtml).join("<br>");
    body.appendChild(div);
  } else {
    const div = document.createElement("div");
    div.className = "alert ok";
    div.innerHTML = "<b>校验通过</b>，通道图同经纬度、同网格，可进行营养诊断。";
    body.appendChild(div);
    state.validated = true;
  }
  if (data.warnings.length) {
    const div = document.createElement("div");
    div.className = "alert warn";
    div.innerHTML = data.warnings.map(escapeHtml).join("<br>");
    body.appendChild(div);
  }

  const files = data.report.files || {};
  const table = document.createElement("table");
  table.className = "meta-table";
  table.innerHTML = `<thead><tr>
    <th>角色</th><th>文件</th><th>尺寸</th><th>波段</th><th>数据类型</th><th>大小</th>
  </tr></thead><tbody></tbody>`;
  const tbody = table.querySelector("tbody");
  for (const role of ["preview", "B1", "B2", "B3", "B4"]) {
    const f = files[role];
    if (!f) continue;
    const tr = document.createElement("tr");
    const meta = f.meta;
    tr.innerHTML = `
      <td>${f.role_label}</td>
      <td title="${f.name}">${f.name}</td>
      <td>${meta ? meta.width + " × " + meta.height : "读取失败"}</td>
      <td>${meta ? meta.count : "-"}</td>
      <td>${meta ? meta.dtype : "-"}</td>
      <td>${fmtSize(f.size)}</td>`;
    tbody.appendChild(tr);
  }
  body.appendChild(table);
  updateRunButton();
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

/* ---------------- 参数设置 ---------------- */

/* 分级提示词：第 1 级固定“最差”，最后一级固定“最好/最旺”。
   2–5 级按常用叫法给出每级推荐；6 级及以上只推荐首尾两级，中间不做提示。 */
function levelHintLabels(count) {
  if (count === 2) return ["最差", "最好/最旺"];
  if (count === 3) return ["最差", "一般", "最好"];
  if (count === 4) return ["最差", "较差", "较好", "最好/最旺"];
  if (count === 5) return ["最差", "较差", "一般", "较好", "最好/最旺"];
  const labels = new Array(count).fill("");
  labels[0] = "最差";
  labels[count - 1] = "最好/最旺";
  return labels;
}

/* 默认累计占比（%）：用户指定的推荐值；6 级及以上只定最后一级为 100，其余留空。 */
function defaultQuantiles(count) {
  const presets = {
    2: [50, 100],
    3: [25, 75, 100],
    4: [15, 50, 85, 100],
    5: [10, 30, 70, 90, 100],
  };
  if (presets[count]) return presets[count];
  const quants = new Array(count).fill(null);
  quants[count - 1] = 100;
  return quants;
}

/* 分级默认配色：与后端 build_class_legend 一致（红=差 -> 绿=好，4/5 级用固定配色）。 */
const CLASS_PALETTE_4 = ["#d73027", "#fee08b", "#a6d96a", "#1a9850"];
const CLASS_PALETTE_5 = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#1a9850"];

function legendColor(levelCount, index) {
  const count = Math.max(2, levelCount);
  const hexToRgb = (hex) => {
    const h = hex.replace("#", "");
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  };
  const rgbToHex = (r, g, b) =>
    "#" + [r, g, b].map((v) => Math.round(Math.min(255, Math.max(0, v))).toString(16).padStart(2, "0")).join("");
  if (count === 4) return CLASS_PALETTE_4[index];
  if (count === 5) return CLASS_PALETTE_5[index];
  const stopsPos = [0.0, 0.25, 0.5, 0.75, 1.0];
  const stopColors = CLASS_PALETTE_5.map(hexToRgb);
  const pos = count === 1 ? 0 : index / (count - 1);
  let seg = 0;
  while (seg < stopsPos.length - 2 && pos > stopsPos[seg + 1]) seg++;
  const t = (pos - stopsPos[seg]) / (stopsPos[seg + 1] - stopsPos[seg]);
  const rgb = stopColors[seg].map((v, ch) => v + (stopColors[seg + 1][ch] - v) * t);
  return rgbToHex(rgb[0], rgb[1], rgb[2]);
}

function readLevelTable() {
  const rows = [];
  $("level-table").querySelectorAll("tbody tr").forEach((tr) => {
    rows.push({
      level: parseInt(tr.dataset.level, 10),
      quantile: parseFloat(tr.querySelector(".q-input").value),
      fertilizer: tr.querySelector(".fert-input").value,
      label: tr.querySelector(".hint-input").value,
      color: tr.dataset.color || "#1a9850",
    });
  });
  return rows;
}

/* 由累计占比（%）计算每级占比（%）：本级累计 - 上级累计，首级即其累计值。 */
function levelPercents(rows) {
  return rows.map((r, i) => {
    if (isNaN(r.quantile)) return null;
    const prev = i === 0 ? 0 : rows[i - 1].quantile;
    if (isNaN(prev)) return null;
    return Math.max(0, +(r.quantile - prev).toFixed(2));
  });
}

function renderLevelTable(resetQuantiles, seed) {
  const stage = currentStage();
  const table = $("level-table");
  const count = parseInt($("level-count").value, 10);
  let oldRows = resetQuantiles ? [] : readLevelTable();
  if (resetQuantiles && seed) {
    oldRows = (seed.quantiles || []).map((q, i) => ({
      level: i + 1,
      quantile: q,
      fertilizer: seed.manual ? (seed.manual[i] ?? "") : "",
      label: seed.labels ? (seed.labels[i] ?? "") : "",
    }));
  }
  const hints = levelHintLabels(count);
  const defaults = defaultQuantiles(count);
  const formulaMapping = computeFormulaMapping(count);

  table.innerHTML = `<thead><tr>
    <th>等级</th><th>长势</th><th>累计占比%</th><th>施肥量</th><th>占比</th>
  </tr></thead>`;
  const tbody = document.createElement("tbody");
  for (let i = 0; i < count; i++) {
    const level = i + 1;
    const old = oldRows.find((r) => r.level === level);
    const defaultQ = defaults[i];
    /* 末行强制 100%，其余优先保留用户已填值，否则用默认推荐值 */
    const quantile = i === count - 1 ? 100 : (old && !isNaN(old.quantile) ? old.quantile : defaultQ);
    const color = (old && old.color) || legendColor(count, i);
    const manual = document.querySelector('input[name="mapping-mode"]:checked').value === "manual";
    const fertValue = old && old.fertilizer !== "" ? old.fertilizer : formulaMapping[level];
    const seedLabel = old && old.label != null ? old.label : "";
    const tr = document.createElement("tr");
    tr.dataset.level = level;
    tr.dataset.color = color;
    tr.innerHTML = `
      <td class="level-name"><button type="button" class="color-swatch" data-color="${color}" title="点击更换颜色" style="background:${color}"></button>第 ${level} 级</td>
      <td><input type="text" class="hint-input" value="${escapeHtml(seedLabel)}" placeholder="${hints[i]}" title="${hints[i] ? "推荐：" + hints[i] : "无推荐，可自定义"}"></td>
      <td><input type="number" class="q-input" step="0.01" min="0" max="100" value="${quantile == null ? "" : quantile}"></td>
      <td><input type="number" class="fert-input" step="0.01" min="0" value="${fertValue ?? ""}" ${manual ? "" : "disabled"}></td>
      <td class="percent-cell">-</td>`;
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);

  table.querySelectorAll(".q-input").forEach((inp) => {
    inp.addEventListener("change", onQuantileInput);
    inp.addEventListener("input", syncSelectedGroupFromControls);
  });
  table.querySelectorAll(".hint-input, .fert-input").forEach((inp) => {
    inp.addEventListener("input", syncSelectedGroupFromControls);
  });
  table.querySelectorAll(".color-swatch").forEach((btn) => btn.addEventListener("click", onPickColor));
  updatePercentColumn();
}

/* 点击色块：弹出颜色选择器（系统色盘 + 16 进制色号输入）。 */
function onPickColor(event) {
  const swatch = event.currentTarget;
  const tr = swatch.closest("tr");
  let picker = swatch.querySelector('input[type="color"]');
  if (!picker) {
    picker = document.createElement("input");
    picker.type = "color";
    picker.value = swatch.dataset.color || "#1a9850";
    picker.style.position = "absolute";
    picker.style.width = "0";
    picker.style.height = "0";
    picker.style.border = "0";
    picker.style.padding = "0";
    picker.style.opacity = "0";
    swatch.style.position = "relative";
    swatch.appendChild(picker);
    picker.addEventListener("input", () => applyLevelColor(tr, picker.value));
  }
  picker.click();
}

function applyLevelColor(tr, color) {
  if (!/^#[0-9a-fA-F]{6}$/.test(color)) return;
  tr.dataset.color = color;
  const swatch = tr.querySelector(".color-swatch");
  swatch.dataset.color = color;
  swatch.style.background = color;
  if (state.result) renderLegendPreviewIfAny();
}

/* 结果图例/预览若已渲染，同步自定义颜色（仅当级数一致时）。 */
function renderLegendPreviewIfAny() {
  const rows = readLevelTable();
  if (!state.result || !state.result.level_stats) return;
  if (state.result.level_stats.length !== rows.length) return;
  state.result.level_stats.forEach((s, i) => { s.color = rows[i].color; });
  const table = $("result-table");
  table.querySelectorAll("tbody tr").forEach((tr, i) => {
    const swatch = tr.querySelector(".legend-swatch");
    const bar = tr.querySelector(".percent-bar-fill");
    if (swatch) swatch.style.background = rows[i].color;
    if (bar) bar.style.background = rows[i].color;
  });
  if (typeof renderClassPreviewColors === "function") renderClassPreviewColors();
  renderMapInfo();
}

function updatePercentColumn() {
  const rows = readLevelTable();
  const percents = levelPercents(rows);
  const maxPercent = Math.max(...percents.filter((p) => p != null), 0);
  const table = $("level-table");
  table.querySelectorAll("tbody tr").forEach((tr, i) => {
    const cell = tr.querySelector(".percent-cell");
    const p = percents[i];
    if (p == null) {
      cell.innerHTML = '<span class="percent-na">-</span>';
      return;
    }
    const barW = maxPercent > 0 ? (p / maxPercent * 100).toFixed(1) : 0;
    cell.innerHTML = `
      <span class="percent-text">${p.toFixed(2)}%</span>
      <div class="percent-bar"><div class="percent-bar-fill" style="width:${barW}%;background:${tr.dataset.color}"></div></div>`;
  });
}

function onQuantileInput() {
  /* 失焦时把最后一级占比自动补足为 100（分位断点要求），输入过程中不强行改写。 */
  const rows = readLevelTable();
  const last = rows[rows.length - 1];
  if (isNaN(last.quantile) || last.quantile !== 100) {
    $("level-table").querySelector("tbody tr:last-child .q-input").value = 100;
  }
  updatePercentColumn();
  renderStageFormula();
  syncSelectedGroupFromControls();
}

function setFertInputsEditable() {
  const manual = document.querySelector('input[name="mapping-mode"]:checked').value === "manual";
  $("level-table").querySelectorAll(".fert-input").forEach((inp) => { inp.disabled = !manual; });
}

function initParams() {
  const grid = $("stage-grid");
  for (const stage of state.config.stages) {
    const item = document.createElement("div");
    item.className = "stage-item" + (stage.id === state.stageId ? " selected" : "");
    item.dataset.id = stage.id;
    item.innerHTML = `${stage.id}. ${stage.name}<small>${stage.index_name}</small>`;
    item.addEventListener("click", () => {
      state.stageId = stage.id;
      grid.querySelectorAll(".stage-item").forEach((el) => el.classList.toggle("selected", el.dataset.id === stage.id));
      /* 切换生育期只更换诊断模型；分级与施肥设置保持当前绑定对象（全局或所选分组） */
      if (roi.selectedGroup != null) {
        applySettingsToControls(settingsOfGroup(roi.groups[roi.selectedGroup]));
      } else {
        $("level-count").value = stage.level_count;
        renderLevelTable(true);
      }
      renderStageFormula();
    });
    grid.appendChild(item);
  }

  const presc = state.config.prescription;
  $("f-base-level").value = presc.formula_mode.base_level;
  $("f-base-fert").value = presc.formula_mode.base_fertilizer;
  $("f-coeff").value = presc.formula_mode.change_coefficient;

  document.querySelectorAll('input[name="mapping-mode"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      const manual = document.querySelector('input[name="mapping-mode"]:checked').value === "manual";
      $("mapping-formula").classList.toggle("hidden", manual);
      setFertInputsEditable();
      if (!manual) syncFormulaToFertInputs();
      syncSelectedGroupFromControls();
    });
  });

  $("level-count").addEventListener("change", () => {
    if (roi.selectedGroup != null) {
      /* 分组模式下改级数：先保存当前行，再按推荐值扩充/收缩该组 */
      syncSelectedGroupFromControls();
      const g = roi.groups[roi.selectedGroup];
      g.level_count = parseInt($("level-count").value, 10);
      const defaultsQ = defaultQuantiles(g.level_count);
      g.quantiles = Array.from({ length: g.level_count }, (_, i) => g.quantiles[i] ?? defaultsQ[i]);
      g.labels = Array.from({ length: g.level_count }, (_, i) => g.labels[i] ?? levelHintLabels(g.level_count)[i]);
      g.manual = {};
      applySettingsToControls(settingsOfGroup(g));
      return;
    }
    renderLevelTable(true);
  });

  ["f-base-level", "f-base-fert", "f-coeff"].forEach((id) => {
    $(id).addEventListener("input", () => {
      syncFormulaToFertInputs();
      syncSelectedGroupFromControls();
    });
  });

  /* 重采样栅格宽度默认值来自服务端配置（用户可在表单中修改，最低 1m） */
  const rs = state.config.resampling;
  if (rs && rs.target_ground_resolution_m) {
    $("resample-cell").value = +rs.target_ground_resolution_m;
  }

  renderStageFormula();
  renderLevelTable();
  $("btn-run").addEventListener("click", doRun);
}

function currentStage() {
  return state.config.stages.find((s) => s.id === state.stageId);
}

function computeFormulaMapping(levelCount) {
  const stage = currentStage();
  const count = levelCount || parseInt($("level-count").value, 10);
  const baseLevel = parseFloat($("f-base-level").value);
  const baseFert = parseFloat($("f-base-fert").value);
  const coeff = parseFloat($("f-coeff").value);
  const digits = state.config.round_digits || 2;
  const mapping = {};
  for (let level = 1; level <= count; level++) {
    mapping[level] = +(baseFert * (1 + (baseLevel - level) * coeff)).toFixed(digits);
  }
  return mapping;
}

function syncFormulaToFertInputs() {
  const formula = computeFormulaMapping();
  $("level-table").querySelectorAll("tbody tr").forEach((tr) => {
    const inp = tr.querySelector(".fert-input");
    const v = formula[parseInt(tr.dataset.level, 10)];
    if (v != null) inp.value = v;
  });
}

function renderStageFormula() {
  const stage = currentStage();
  const el = $("stage-formula");
  el.classList.remove("hidden");
  el.innerHTML = `
    <div>诊断指标：<b>${stage.diagnosis_name}</b> ｜ 指数：${stage.index_formula}</div>
    <div>模型：${stage.diagnosis_formula}</div>`;
}

function readRunParams() {
  const manual = document.querySelector('input[name="mapping-mode"]:checked').value === "manual";
  const makeMode = document.querySelector('input[name="make-mode"]:checked').value;
  const params = {
    stage: state.stageId,
    make_mode: makeMode,
    mapping_mode: manual ? "manual" : "formula",
    formula: {
      base_level: parseFloat($("f-base-level").value),
      base_fertilizer: parseFloat($("f-base-fert").value),
      change_coefficient: parseFloat($("f-coeff").value),
    },
    manual_mapping: {},
    level_thresholds: readLevelTable().map((r) => ({
      level: r.level,
      quantile: r.quantile / 100.0,
      growth_label: $("level-table").querySelector(`tr[data-level="${r.level}"] .hint-input`).value.trim()
        || $("level-table").querySelector(`tr[data-level="${r.level}"] .hint-input`).placeholder,
      color: r.color,
    })),
    resampling: {
      target_ground_resolution_m: parseFloat($("resample-cell").value),
    },
  };
  if (makeMode === "roi") {
    params.roi_regions = roi.regions.map((region) => ({
      hull: region.hull.map((p) => [p.lon, p.lat]),
      points: region.points.map((p) => [p.lon, p.lat]),
    }));
    if (roi.groups.length) {
      params.roi_groups = roi.groups.map((g, gi) => {
        const quantiles = g.quantiles.slice(0, g.level_count);
        if (quantiles[quantiles.length - 1] !== 100) quantiles[quantiles.length - 1] = 100;
        return {
          name: `组${gi + 1}`,
          region_indexes: g.regions.slice().sort((a, b) => a - b),
          level_count: g.level_count,
          mapping_mode: g.mode,
          formula: g.mode === "formula"
            ? { base_level: g.formula.b, base_fertilizer: g.formula.qb, change_coefficient: g.formula.k }
            : {},
          manual_mapping: g.mode === "manual"
            ? Object.fromEntries(Object.entries(g.manual).map(([lv, v]) => [lv, parseFloat(v)]))
            : {},
          thresholds: quantiles.map((q, i) => ({
            level: i + 1,
            quantile: q / 100,
            growth_label: g.labels[i] || "",
            color: legendColor(g.level_count, i),
          })),
        };
      });
    }
  }
  if (manual) {
    readLevelTable().forEach((r) => {
      const v = parseFloat(r.fertilizer);
      if (!isNaN(v)) params.manual_mapping[r.level] = v;
    });
  }
  return params;
}

function updateRunButton() {
  const s = state.session;
  const files = (s && s.files) || {};
  const ready = state.validated && files.B1 && files.B2 && files.B3 && files.B4;
  $("btn-run").disabled = !ready;
}

/* ---------------- 运行 ---------------- */
async function doRun() {
  if (state.running) { toast("已有任务正在运行", true); return; }
  if (roi.editing) {
    toast("请先退出感兴趣区域编辑模式，再开始生成", true);
    return;
  }
  const params = readRunParams();
  const manual = params.mapping_mode === "manual";
  if (manual && Object.keys(params.manual_mapping).length !== params.level_thresholds.length) {
    toast("请填写所有等级的施肥量", true);
    return;
  }
  const quantiles = params.level_thresholds.map((t) => t.quantile);
  if (quantiles.some((q) => isNaN(q) || q <= 0 || q > 1)) {
    toast("各级累计占比必须是 0–100 之间的数值", true);
    return;
  }
  for (let i = 1; i < quantiles.length; i++) {
    if (quantiles[i] <= quantiles[i - 1]) {
      toast("累计占比必须从第 1 级到最后一级严格递增", true);
      return;
    }
  }
  if (quantiles[quantiles.length - 1] !== 1) {
    params.level_thresholds[params.level_thresholds.length - 1].quantile = 1;
  }
  const cellSize = params.resampling.target_ground_resolution_m;
  if (isNaN(cellSize) || cellSize < 1) {
    toast("栅格宽度必须是 ≥ 1 的数值", true);
    return;
  }
  if (params.make_mode === "roi") {
    if (!params.roi_regions || !params.roi_regions.length) {
      toast("感兴趣区域为空，请先在预览图上标点", true);
      return;
    }
    if (roi.current.length) {
      toast("有未完成的标点区域未计入（需右键完成才算一个区域）", true);
    }
    for (const g of params.roi_groups || []) {
      if (!g.region_indexes.length) {
        toast(`分组「${g.name}」未勾选任何区域`, true);
        return;
      }
      const qs = g.thresholds.map((t) => t.quantile * 100);
      for (let i = 0; i < qs.length; i++) {
        if (isNaN(qs[i]) || qs[i] <= 0 || qs[i] > 100) {
          toast(`分组「${g.name}」第 ${i + 1} 级累计占比必须是 0–100 之间的数值`, true);
          return;
        }
        if (i > 0 && qs[i] <= qs[i - 1]) {
          toast(`分组「${g.name}」各级累计占比必须从小到大严格递增`, true);
          return;
        }
      }
      if (g.mapping_mode === "manual") {
        for (let lv = 1; lv <= g.level_count; lv++) {
          if (isNaN(parseFloat(g.manual_mapping[lv]))) {
            toast(`分组「${g.name}」请填写第 ${lv} 级的施肥量`, true);
            return;
          }
        }
      }
    }
  }
  try {
    $("card-progress").classList.remove("hidden");
    $("card-progress").scrollIntoView({ behavior: "smooth", block: "nearest" });
    await api("/api/run", params);
    state.running = true;
    $("progress-fill").style.width = "0%";
    pollJob();
  } catch (err) {
    toast(err.message, true);
  }
}

function pollJob() {
  if (state.pollTimer) clearTimeout(state.pollTimer);
  const tick = async () => {
    try {
      const data = await api("/api/job");
      const job = data.job;
      if (!job) { state.pollTimer = setTimeout(tick, 800); return; }
      renderProgress(job);
      if (job.running) {
        state.pollTimer = setTimeout(tick, 700);
      } else {
        state.running = false;
        onJobFinished(job);
      }
    } catch (err) {
      state.pollTimer = setTimeout(tick, 1500);
    }
  };
  tick();
}

function renderProgress(job) {
  $("progress-fill").style.width = (job.progress || 0) + "%";
  $("progress-percent").textContent = (job.progress || 0).toFixed ? job.progress.toFixed(1) + "%" : job.progress + "%";
  $("progress-phase").textContent = job.phase_label || "-";
  $("progress-message").textContent = job.message || "";
  const log = $("log-output");
  log.textContent = (job.log || []).join("\n");
  log.scrollTop = log.scrollHeight;
}

async function onJobFinished(job) {
  renderProgress(job);
  if (job.error) {
    toast("任务失败：" + job.error, true);
    return;
  }
  const result = job.result || {};
  if (result.level_stats) {
    const data = await api("/api/results");
    state.result = data.result;
    state.annotated = {};
    renderResults(state.result);
  } else if (result.dj2pr) {
    renderDj2prReport(result.dj2pr);
  } else if (result.input_preview) {
    await refreshSession();
    showPreview("input");
    toast("整体预览图已生成");
  }
}

/* ---------------- 结果 ---------------- */
function hideResultCards() {
  ["card-validation", "card-params", "card-progress", "card-result"].forEach((id) => $(id).classList.add("hidden"));
  $("btn-run").disabled = true;
  setViewerTabEnabled(false);
}

function levelRowHtml(s, maxCount) {
  const range = (s.min == null ? "−∞" : formatNum(s.min)) + " ~ " + (s.max == null ? "+∞" : formatNum(s.max));
  return `
    <tr>
      <td><span class="legend-swatch" style="background:${s.color}"></span> 第 ${s.level} 级</td>
      <td>${escapeHtml(s.growth_label || "")}</td>
      <td>${range}</td>
      <td><b>${s.fertilizer}</b></td>
      <td>${s.count.toLocaleString()}</td>
      <td>${s.percent}%<div style="height:5px;background:#e5ece5;border-radius:3px;margin-top:3px">
        <div style="height:100%;width:${(s.count / maxCount * 100).toFixed(1)}%;background:${s.color};border-radius:3px"></div></div></td>
    </tr>`;
}

function renderResults(result) {
  $("card-result").classList.remove("hidden");
  state.validated = true;
  updateRunButton();

  const groups = result.group_results && result.group_results.length ? result.group_results : null;
  const chips = $("result-chips");
  chips.innerHTML = `
    <span class="chip">${result.stage.name} · ${result.stage.index_name} 模型</span>
    <span class="chip">${result.make_mode === "roi" ? `感兴趣区域制作（${result.roi_region_count} 个区域）` : "全图制作"}</span>
    ${groups ? `<span class="chip">区域分组 ${groups.length} 组</span>` : ""}
    <span class="chip">有效像素 ${result.valid_pixel_count.toLocaleString()}</span>
    <span class="chip">耗时 ${result.elapsed_seconds}s</span>
    <span class="chip">映射：${groups ? "按分组设置" : (result.mapping_mode === "manual" ? "手动指定" : "公式计算")}</span>`;

  const table = $("result-table");
  if (groups) {
    let html = "";
    for (const g of groups) {
      const maxCount = Math.max(...g.level_stats.map((s) => s.count), 1);
      html += `<thead><tr><th colspan="6" class="group-head">
        ${escapeHtml(g.name)}（区域 ${g.region_numbers.join("、")} · ${(g.area && g.area.area_mu || 0).toFixed(2)} 亩 ·
        ${g.level_count} 级 · ${g.mapping_mode === "manual" ? "手动指定" : "公式计算"}）
      </th></tr>
      <tr><th>等级</th><th>长势/产量</th><th>诊断值区间</th><th>施肥量 (kg/亩)</th><th>像元数</th><th>占比</th></tr></thead><tbody>
      ${g.level_stats.map((s) => levelRowHtml(s, maxCount)).join("")}
      </tbody>`;
    }
    table.innerHTML = html;
  } else {
    const maxCount = Math.max(...result.level_stats.map((s) => s.count), 1);
    table.innerHTML = `<thead><tr>
      <th>等级</th><th>长势/产量</th><th>诊断值区间</th><th>施肥量 (kg/亩)</th><th>像元数</th><th>占比</th>
    </tr></thead><tbody>
    ${result.level_stats.map((s) => levelRowHtml(s, maxCount)).join("")}
    </tbody>`;
  }

  $("legend-note").textContent = groups
    ? "分组独立分级与映射：" + groups.map((g) => `${g.name} ${JSON.stringify(g.mapping)}`).join("；") + "，单位 " + result.unit
    : "分级区间按诊断值分位数计算（min 开区间、max 闭区间）；施肥量映射 " +
      JSON.stringify(result.mapping) + " " + result.unit;

  const filesEl = $("output-files");
  filesEl.innerHTML = "";
  for (const f of result.output_files) {
    const row = document.createElement("div");
    row.className = "output-file";
    row.innerHTML = `
      <span class="of-name" title="${f.path}">${f.name}</span>
      <span class="of-size">${fmtSize(f.size)}</span>
      <a href="/api/download?path=${encodeURIComponent(f.path)}">下载</a>`;
    filesEl.appendChild(row);
  }

  setViewerTabEnabled(true);
  showPreview("prescription");
}

function formatNum(v) {
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}

/* ---------------- 图窗 ---------------- */
const viewer = { scale: 1, tx: 0, ty: 0, naturalW: 0, naturalH: 0, url: null };

function resetViewer() {
  $("viewer-img").removeAttribute("src");
  $("viewer-transform").style.display = "none";
  $("viewer-empty").style.display = "flex";
  $("viewer-legend").innerHTML = "";
  $("viewer-stats").innerHTML = "";
  const mi = $("viewer-mapinfo");
  mi.classList.add("hidden");
  mi.innerHTML = "";
  viewer.url = null;
  document.querySelectorAll("#viewer-tabs .tab").forEach((t, i) => {
    t.classList.toggle("active", i === 0);
  });
  state.currentView = "input";
}

function setViewerTabEnabled(enabled) {
  document.querySelectorAll("#viewer-tabs .tab").forEach((t) => {
    if (t.dataset.view !== "input") t.disabled = !enabled;
  });
}

function showPreview(view) {
  if (view === "input") {
    if (!state.session || !state.session.has_input_preview) {
      toast("预览图尚未生成", true);
      return;
    }
    setViewerImage("/previews/input_preview.jpg?t=" + Date.now(), "none");
  } else if (state.result) {
    let url = view === "class" ? "/previews/class_preview.png" : "/previews/prescription_preview.png";
    let bust = state.result.run_id;
    const annotated = state.annotated && state.annotated[view];
    if (annotated) {
      url = annotated.url;
      bust = Date.now();
    }
    setViewerImage(url + "?t=" + bust, view);
  } else {
    return;
  }
  state.currentView = view;
  document.querySelectorAll("#viewer-tabs .tab").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
  renderViewerStats();
  renderMapInfo();
  syncLeftColumnHeight();
}

function setViewerImage(url, legendMode) {
  const img = $("viewer-img");
  const canvas = $("roi-canvas");
  const isInputView = legendMode === "none";
  canvas.classList.toggle("hidden", !isInputView);
  img.onload = () => {
    viewer.naturalW = img.naturalWidth;
    viewer.naturalH = img.naturalHeight;
    $("viewer-transform").style.display = "block";
    $("viewer-empty").style.display = "none";
    if (isInputView) {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
    }
    fitViewer();
  };
  img.src = url;
  renderLegend(legendMode);
}

function renderLegend(mode) {
  const el = $("viewer-legend");
  if (mode === "none") {
    el.innerHTML = `<span>整体预览图（仅展示，不参与诊断计算）</span>`;
  } else if (mode === "class" || mode === "prescription") {
    /* 长势等级 / 推荐施肥量图例已移至图面注记模块（viewer-mapinfo） */
    el.innerHTML = "";
  } else {
    el.innerHTML = "";
  }
}

/* 预览图下方的统计数据（随当前视图切换） */
function formatMuArea(area) {
  if (!area) return "-";
  const m2 = area.area_m2;
  const m2Text = m2 >= 100 ? Math.round(m2).toLocaleString() : m2.toFixed(1);
  return `${area.area_mu.toFixed(2)} 亩（${m2Text} m²）`;
}

function statsRow(items) {
  return items
    .map(([label, value]) =>
      `<span class="vstat"><span class="vstat-label">${label}</span><span class="vstat-value">${value}</span></span>`)
    .join("");
}

function renderViewerStats() {
  const el = $("viewer-stats");
  const view = state.currentView;

  if (view === "prescription" && state.result && state.result.prescription_stats) {
    const s = state.result.prescription_stats;
    el.innerHTML = statsRow([
      ["施肥总量", `${s.total_fertilizer_kg.toFixed(2)} kg`],
      ["处方图面积", formatMuArea(s)],
      ["平均每亩施肥量", `${s.avg_fertilizer_kg_per_mu.toFixed(2)} kg/亩`],
    ]);
  } else if (view === "class" && state.result && state.result.diagnosis_value_range) {
    const r = state.result.diagnosis_value_range;
    if (r.min == null || r.max == null) { el.innerHTML = ""; return; }
    const name = state.result.stage.diagnosis_name;
    el.innerHTML = statsRow([["诊断值总区间", `${formatNum(r.min)} ~ ${formatNum(r.max)}（${name}）`]]);
  } else if (view === "input") {
    const items = [];
    if (state.session && state.session.input_area) {
      items.push(["总面积", formatMuArea(state.session.input_area)]);
    }
    const roiArea = roi.area || (state.result && state.result.roi_area);
    if (roiArea && (roiArea.area_mu > 0 || roiArea.area_m2 > 0)) {
      items.push(["感兴趣区域面积", formatMuArea(roiArea)]);
    }
    el.innerHTML = items.length ? statsRow(items) : "";
  } else {
    el.innerHTML = "";
  }
}

/* 图面注记（田块面积/制图单位/制图时间，可编辑）+ 长势等级/推荐施肥量图例模块 */
const MAPINFO_KEY = "prescription.mapInfo";

function loadMapInfo() {
  try { return JSON.parse(localStorage.getItem(MAPINFO_KEY)) || {}; } catch (e) { return {}; }
}

function saveMapInfo(info) {
  try { localStorage.setItem(MAPINFO_KEY, JSON.stringify(info)); } catch (e) { /* ignore */ }
}

function formatMuPlain(v) {
  if (v == null || isNaN(v)) return "";
  return `${Math.round(v * 100) / 100}亩`;
}

/* 田块面积的自动默认值：处方图面积 > ROI 面积 > 影像总面积 */
function computedFieldArea() {
  const r = state.result;
  if (r && r.prescription_stats && r.prescription_stats.area_mu != null) return r.prescription_stats.area_mu;
  if (r && r.roi_area && r.roi_area.area_mu > 0) return r.roi_area.area_mu;
  const s = state.session;
  if (s && s.input_area && s.input_area.area_mu != null) return s.input_area.area_mu;
  return null;
}

function mapInfoDate() {
  const r = state.result;
  if (r && r.run_id) {
    const m = /^(\d{4})(\d{2})(\d{2})/.exec(r.run_id);
    if (m) return `${parseInt(m[1], 10)}年${parseInt(m[2], 10)}月${parseInt(m[3], 10)}日`;
  }
  const now = new Date();
  return `${now.getFullYear()}年${now.getMonth() + 1}月${now.getDate()}日`;
}

function renderMapInfo() {
  const el = $("viewer-mapinfo");
  const view = state.currentView;
  if ((view !== "class" && view !== "prescription") || !state.result) {
    el.classList.add("hidden");
    el.innerHTML = "";
    return;
  }
  const info = loadMapInfo();
  const areaDefault = computedFieldArea();
  const areaDefaultText = areaDefault != null ? formatMuPlain(areaDefault) : "";
  /* 田块面积的手动修改按 run 记录，避免上一地块的数值带到新成果里 */
  const areaByRun = info.areaByRun || {};
  const areaUser = state.result.run_id ? areaByRun[state.result.run_id] : null;
  const areaText = areaUser != null && areaUser !== "" ? areaUser : areaDefaultText;
  const unitText = info.unit != null && info.unit !== "" ? info.unit : "扬州大学";
  const dateText = info.date != null && info.date !== "" ? info.date : mapInfoDate();

  let legendHtml = "";
  if (view === "class" && state.result.level_stats) {
    const itemsHtml = (stats) => stats.map((lv) =>
      `<span class="mi-item"><span class="mi-swatch" style="background:${lv.color}"></span>${lv.level}级 ${escapeHtml(lv.growth_label || "")}</span>`).join("");
    const groups = state.result.group_results && state.result.group_results.length ? state.result.group_results : null;
    let body;
    if (groups) {
      /* 各分组等级数量与每级颜色、长势标签完全一致时，图例融合为单一列表（与嵌入样式一致） */
      const base = groups[0].level_stats;
      const identical = base.length > 0 && groups.every((g) => g.level_stats.length === base.length) &&
        groups.every((g) => g.level_stats.every((s, i) =>
          s.color === base[i].color && (s.growth_label || "") === (base[i].growth_label || "")));
      body = identical
        ? `<span class="mi-items">${itemsHtml(base)}</span>`
        : groups.map((g) => `<div class="mi-group"><b>${escapeHtml(g.name)}</b><span class="mi-items">${itemsHtml(g.level_stats)}</span></div>`).join("");
    } else {
      body = `<span class="mi-items">${itemsHtml(state.result.level_stats)}</span>`;
    }
    legendHtml = `<div class="mi-legend"><div class="mi-legend-title">长势等级</div>${body}</div>`;
  } else if (view === "prescription" && state.result.prescription_preview_range) {
    const range = state.result.prescription_preview_range;
    legendHtml = `
      <div class="mi-legend">
        <div class="mi-legend-title">推荐施肥量</div>
        <div class="mi-gradient">
          <span>${formatNum(range.min)}</span>
          <span class="gradient-bar" style="background:${state.result.gradient_css}"></span>
          <span>${formatNum(range.max)} kg/亩</span>
        </div>
      </div>`;
  }

  el.innerHTML = `
    <div class="mapinfo-card">
      <div class="mi-fields">
        <div class="mi-line"><label>田块面积：</label><input class="mi-input mi-area" value="${escapeHtml(String(areaText))}" placeholder="${escapeHtml(areaDefaultText)}" title="点击编辑，清空则显示自动计算的面积"></div>
        <div class="mi-line"><label>制图单位：</label><input class="mi-input mi-unit" value="${escapeHtml(unitText)}" placeholder="如：扬州大学"></div>
        <div class="mi-line"><label>制图时间：</label><input class="mi-input mi-date" value="${escapeHtml(dateText)}" placeholder="如：2025年8月7日"></div>
      </div>
      ${legendHtml}
      <div class="mi-actions">
        <button class="btn tiny" type="button" data-act="embed" title="把注记与图例绘制到预览图上">🖼 嵌入预览图</button>
        <button class="btn tiny" type="button" data-act="download" title="下载嵌入注记后的预览图">⬇ 下载</button>
      </div>
    </div>`;
  el.classList.remove("hidden");

  el.querySelector(".mi-area").addEventListener("input", (e) => {
    if (state.result && state.result.run_id) {
      areaByRun[state.result.run_id] = e.target.value;
      info.areaByRun = areaByRun;
    }
    saveMapInfo(info);
  });
  el.querySelector(".mi-unit").addEventListener("input", (e) => { info.unit = e.target.value; saveMapInfo(info); });
  el.querySelector(".mi-date").addEventListener("input", (e) => { info.date = e.target.value; saveMapInfo(info); });

  async function ensureAnnotated() {
    const data = await api("/api/preview/annotate", {
      view, area: areaText, unit: unitText, date: dateText,
    });
    state.annotated[view] = data;
    return data;
  }
  el.querySelector('[data-act="embed"]').addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    btn.disabled = true;
    try {
      await ensureAnnotated();
      toast("已把注记嵌入预览图");
      showPreview(view);
    } catch (err) { toast(err.message, true); }
    btn.disabled = false;
  });
  el.querySelector('[data-act="download"]').addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    btn.disabled = true;
    try {
      const data = state.annotated[view] || await ensureAnnotated();
      const a = document.createElement("a");
      a.href = data.download_url;
      a.download = "";
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) { toast(err.message, true); }
    btn.disabled = false;
  });
}

/* 左栏高度同步：当右侧出现额外信息条（诊断图/处方图注记卡片）或 ROI 列表
   展开使右栏变高时，把左栏拉升到与右栏齐平（内容更长时用自身滑条滚动） */
function syncLeftColumnHeight() {
  const wf = document.querySelector(".workflow");
  const col = document.querySelector(".viewer-column");
  if (!wf || !col) return;
  const infoVisible = !$("viewer-mapinfo").classList.contains("hidden");
  if (infoVisible) {
    /* 显式高度需同时解除 CSS 最小高度限制，右栏不足一屏时也能齐平 */
    wf.style.height = col.offsetHeight + "px";
    wf.style.minHeight = "0";
  } else {
    wf.style.height = "";
    wf.style.minHeight = "";
  }
}

function fitViewer() {
  const stage = $("viewer-stage");
  if (!viewer.naturalW) return;
  const scale = Math.min(
    stage.clientWidth / viewer.naturalW,
    stage.clientHeight / viewer.naturalH
  );
  viewer.scale = scale;
  viewer.tx = (stage.clientWidth - viewer.naturalW * scale) / 2;
  viewer.ty = (stage.clientHeight - viewer.naturalH * scale) / 2;
  applyViewerTransform();
}

function applyViewerTransform() {
  $("viewer-transform").style.transform = `translate(${viewer.tx}px, ${viewer.ty}px) scale(${viewer.scale})`;
  drawRoi();
}

/* ---------------- 感兴趣区域（ROI）编辑 ---------------- */
const roi = { editing: false, regions: [], current: [], area: null, groups: [],
              groupEditing: false, selectedGroup: null, globalSnapshot: null, groupPick: new Set() };

/* 读取当前控件中的分级与施肥设置（全局快照与分组值共用同一形态） */
function captureControlSettings() {
  const rows = readLevelTable();
  return {
    level_count: rows.length,
    mode: document.querySelector('input[name="mapping-mode"]:checked').value,
    formula: { b: $("f-base-level").value, qb: $("f-base-fert").value, k: $("f-coeff").value },
    quantiles: rows.map((r) => r.quantile),
    labels: rows.map((r) => r.label),
    manual: rows.map((r) => r.fertilizer),
  };
}

function settingsOfGroup(g) {
  return {
    level_count: g.level_count,
    mode: g.mode,
    formula: { ...g.formula },
    quantiles: g.quantiles.slice(),
    labels: g.labels.slice(),
    manual: g.quantiles.slice().map((_, i) => g.manual[i + 1] ?? ""),
  };
}

/* 把控件中的当前值写回当前选中的分组 */
function syncSelectedGroupFromControls() {
  if (roi.selectedGroup == null) return;
  const g = roi.groups[roi.selectedGroup];
  if (!g) return;
  const s = captureControlSettings();
  g.level_count = s.level_count;
  g.mode = s.mode;
  g.formula = { ...s.formula };
  g.quantiles = s.quantiles.slice();
  g.labels = s.labels.slice();
  g.manual = {};
  s.manual.forEach((v, i) => { g.manual[i + 1] = v; });
}

/* 把设置（全局快照或某分组）加载进主表格控件 */
function applySettingsToControls(s) {
  $("level-count").value = String(s.level_count);
  document.querySelector(`input[name="mapping-mode"][value="${s.mode}"]`).checked = true;
  $("mapping-formula").classList.toggle("hidden", s.mode === "manual");
  $("f-base-level").value = s.formula.b;
  $("f-base-fert").value = s.formula.qb;
  $("f-coeff").value = s.formula.k;
  setFertInputsEditable();
  renderLevelTable(true, s);
  renderStageFormula();
}

/* 选择要编辑的设置对象：null = 全局（未分组区域），数字 = roi.groups 下标 */
function selectGroup(idx) {
  if (roi.selectedGroup === idx) return;
  if (roi.selectedGroup == null) {
    roi.globalSnapshot = captureControlSettings();
  } else {
    syncSelectedGroupFromControls();
  }
  roi.selectedGroup = idx;
  const s = idx == null
    ? (roi.globalSnapshot || captureControlSettings())
    : (roi.groups[idx] ? settingsOfGroup(roi.groups[idx]) : null);
  if (s) applySettingsToControls(s);
  updateSettingsBinding();
  updateRoiPanel();
}

function updateSettingsBinding() {
  const el = $("settings-binding");
  if (roi.selectedGroup == null) {
    el.classList.add("hidden");
    el.innerHTML = "";
    return;
  }
  const g = roi.groups[roi.selectedGroup];
  if (!g) { el.classList.add("hidden"); return; }
  const displayName = g.name || `组${roi.selectedGroup + 1}`;
  const regions = g.regions.length ? "含 " + g.regions.map((i) => "区域" + (i + 1)).join("、") : "未勾选区域";
  el.classList.remove("hidden");
  el.innerHTML = `正在编辑分组「${escapeHtml(displayName)}」（${regions}）的分级与施肥映射 · ` +
    `<a href="javascript:void(0)" id="link-back-global">返回全局设置</a>`;
  el.querySelector("#link-back-global").addEventListener("click", () => selectGroup(null));
}

async function refreshRoiArea() {
  /* 区域变化后即时计算地面面积（显示在整体预览图下方） */
  roi.area = null;
  if (roi.regions.length && state.session && state.session.has_input_preview) {
    try {
      const data = await api("/api/roi/area", {
        regions: roi.regions.map((r) => ({ hull: r.hull.map((p) => [p.lon, p.lat]) })),
      });
      roi.area = { area_m2: data.area_m2, area_mu: data.area_mu };
    } catch (err) { /* 面积计算失败不打断编辑；运行结果中会重算 */ }
  }
  renderViewerStats();
}

/* 区域删除后重排分组里的区域编号（removedIdx 为被删除的区域下标） */
function remapGroupsAfterRegionDelete(removedIdx) {
  let changed = false;
  roi.groups.forEach((g) => {
    const next = g.regions
      .filter((i) => i !== removedIdx)
      .map((i) => (i > removedIdx ? i - 1 : i));
    if (next.length !== g.regions.length || next.some((v, i) => v !== g.regions[i])) {
      g.regions = next;
      changed = true;
    }
  });
  /* 分组勾选中的区域编号同步重映射，避免指向错位的区域 */
  const nextPick = new Set();
  roi.groupPick.forEach((i) => { if (i !== removedIdx) nextPick.add(i > removedIdx ? i - 1 : i); });
  roi.groupPick = nextPick;
  if (changed) updateRoiPanel();
}

function resetRoi() {
  roi.editing = false;
  roi.regions = [];
  roi.current = [];
  roi.area = null;
  roi.groups = [];
  roi.groupEditing = false;
  roi.selectedGroup = null;
  roi.globalSnapshot = null;
  roi.groupPick = new Set();
  updateSettingsBinding();
  $("viewer-stage").classList.remove("editing");
  $("roi-toolbar").classList.add("hidden");
  $("roi-panel").classList.add("hidden");
  $("roi-canvas").classList.add("hidden");
  $("btn-roi-edit").textContent = "✏️ 编辑感兴趣区域";
  $("btn-roi-edit").classList.remove("primary");
  updateRoiControls();
  updateRoiPanel();
  drawRoi();
  renderViewerStats();
}

function previewLonLat(x, y) {
  const g = state.session && state.session.georef;
  if (!g) return null;
  const sx = (x + 0.5) * g.source_width / g.preview_width;
  const sy = (y + 0.5) * g.source_height / g.preview_height;
  const t = g.transform;
  return { lon: t[0] * sx + t[1] * sy + t[2], lat: t[3] * sx + t[4] * sy + t[5] };
}

function convexHull(points) {
  if (points.length < 3) return points.slice();
  const p = points.slice().sort((a, b) => a.x - b.x || a.y - b.y);
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower = [];
  for (const pt of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], pt) <= 0) lower.pop();
    lower.push(pt);
  }
  const upper = [];
  for (let i = p.length - 1; i >= 0; i--) {
    const pt = p[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], pt) <= 0) upper.pop();
    upper.push(pt);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function pointInPolygon(pt, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x, yi = poly[i].y, xj = poly[j].x, yj = poly[j].y;
    if ((yi > pt.y) !== (yj > pt.y)) {
      const xIntersect = xi + (pt.y - yi) * (xj - xi) / (yj - yi);
      if (pt.x < xIntersect) inside = !inside;
    }
  }
  return inside;
}

function addRoiPoint(x, y) {
  /* 点击已有标点附近视为删除该标点：凸包按剩余标点重算，从而可以内缩 */
  const hitR = 10 / viewer.scale;
  let hitIdx = -1, hitD = Infinity;
  for (let i = 0; i < roi.current.length; i++) {
    const d = Math.hypot(roi.current[i].x - x, roi.current[i].y - y);
    if (d <= hitR && d < hitD) { hitD = d; hitIdx = i; }
  }
  if (hitIdx >= 0) {
    roi.current.splice(hitIdx, 1);
    toast(`已删除标点 ${hitIdx + 1}（凸包按剩余标点重算）`);
    drawRoi();
    updateRoiPanel();
    return;
  }
  for (let i = 0; i < roi.regions.length; i++) {
    if (pointInPolygon({ x, y }, roi.regions[i].hull)) {
      toast(`标点落在已有区域 ${i + 1} 内部，已忽略`, true);
      return;
    }
  }
  const ll = previewLonLat(x, y);
  roi.current.push({
    x, y,
    lon: ll ? +ll.lon.toFixed(8) : null,
    lat: ll ? +ll.lat.toFixed(8) : null,
  });
  drawRoi();
  updateRoiPanel();
}

function finalizeCurrentRegion() {
  if (!roi.current.length) return;
  if (roi.current.length < 3) {
    toast("当前区域至少需要 3 个标点才能构成凸包", true);
    return;
  }
  const hull = convexHull(roi.current);
  roi.regions.push({ points: roi.current, hull });
  roi.current = [];
  drawRoi();
  updateRoiPanel();
  refreshRoiArea();
  toast(`区域 ${roi.regions.length} 已完成（凸包 ${hull.length} 个顶点），可继续标下一个区域`);
}

function undoRoiPoint() {
  if (roi.current.length) {
    roi.current.pop();
  } else if (roi.regions.length) {
    const last = roi.regions.pop();
    remapGroupsAfterRegionDelete(roi.regions.length);
    /* 撤回已完成区域时恢复其标点为进行中区域，可继续增删标点后重新完成 */
    roi.current = last.points.slice();
    refreshRoiArea();
    toast(`已撤回区域 ${roi.regions.length + 1}，其标点已恢复为当前编辑区域`);
  } else {
    return;
  }
  drawRoi();
  updateRoiPanel();
}

function centroidOf(pts) {
  let x = 0, y = 0;
  for (const p of pts) { x += p.x; y += p.y; }
  return { x: x / pts.length, y: y / pts.length };
}

function drawRoi() {
  const canvas = $("roi-canvas");
  if (canvas.classList.contains("hidden") || !canvas.width) return;
  const ctx = canvas.getContext("2d");
  const s = viewer.scale;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const drawDots = (pts, fill, stroke) => {
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4.5 / s, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.lineWidth = 1.5 / s;
      ctx.strokeStyle = stroke;
      ctx.stroke();
    }
  };
  const tracePoly = (pts) => {
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.closePath();
  };

  roi.regions.forEach((region, idx) => {
    tracePoly(region.hull);
    ctx.fillStyle = "rgba(255, 152, 0, 0.16)";
    ctx.fill();
    ctx.lineWidth = 2.5 / s;
    ctx.strokeStyle = "#ef6c00";
    ctx.stroke();
    drawDots(region.hull, "#fff3e0", "#e65100");
    const c = centroidOf(region.hull);
    ctx.font = `${15 / s}px sans-serif`;
    ctx.fillStyle = "#e65100";
    ctx.textAlign = "center";
    ctx.fillText(`区域${idx + 1}`, c.x, c.y);
  });

  if (roi.current.length) {
    if (roi.current.length >= 3) {
      const hull = convexHull(roi.current);
      tracePoly(hull);
      ctx.fillStyle = "rgba(21, 101, 192, 0.10)";
      ctx.fill();
      ctx.lineWidth = 2 / s;
      ctx.setLineDash([8 / s, 6 / s]);
      ctx.strokeStyle = "#1565c0";
      ctx.stroke();
      ctx.setLineDash([]);
    } else {
      ctx.beginPath();
      ctx.moveTo(roi.current[0].x, roi.current[0].y);
      for (let i = 1; i < roi.current.length; i++) ctx.lineTo(roi.current[i].x, roi.current[i].y);
      ctx.lineWidth = 1.5 / s;
      ctx.strokeStyle = "#1565c0";
      ctx.stroke();
    }
    drawDots(roi.current, "#e3f2fd", "#1565c0");
  }
}

function setRoiEditing(on) {
  if (!on && roi.editing && roi.current.length) {
    const discard = window.confirm("当前区域未完成，是否删除当前编辑区域并退出编辑？");
    if (!discard) return;  // 用户取消，保持编辑状态
    roi.current = [];
    drawRoi();
    updateRoiPanel();
  }
  const img = $("viewer-img");
  if (on && (!img.src || !$("viewer-transform").style.display || $("viewer-transform").style.display === "none")) {
    toast("请先生成整体预览图", true);
    return;
  }
  if (on && !(state.session && state.session.georef)) {
    toast("缺少地理参考信息，无法记录经纬度", true);
    return;
  }
  if (on && state.currentView !== "input") {
    toast("请切换到「整体预览」视图后再编辑", true);
    return;
  }
  roi.editing = on;
  $("viewer-stage").classList.toggle("editing", on);
  $("roi-toolbar").classList.toggle("hidden", !on);
  $("btn-roi-edit").textContent = on ? "✏️ 编辑中..." : "✏️ 编辑感兴趣区域";
  $("btn-roi-edit").classList.toggle("primary", on);
}

function updateRoiControls() {
  const ready = state.session && state.session.has_input_preview && state.session.georef;
  $("btn-roi-edit").disabled = !ready;
  const count = roi.regions.length;
  $("make-mode-roi-count").textContent = count;
  const roiRadio = document.querySelector('input[name="make-mode"][value="roi"]');
  roiRadio.disabled = count === 0;
  if (count === 0) {
    document.querySelector('input[name="make-mode"][value="full"]').checked = true;
  }
  /* 「编辑分组」按钮仅在勾选「感兴趣区域制作」且已有区域时显示 */
  const roiChecked = document.querySelector('input[name="make-mode"]:checked').value === "roi";
  $("btn-roi-groups").classList.toggle("hidden", !(roiChecked && count > 0));
  $("btn-roi-groups").disabled = count === 0;
  const btnGroups = $("btn-roi-groups");
  btnGroups.textContent = roi.groupEditing ? "✔ 完成分组" : "👥 编辑分组";
  btnGroups.classList.toggle("primary", roi.groupEditing);
}

function updateRoiPanel() {
  const panel = $("roi-panel");
  if (!roi.regions.length && !roi.current.length) {
    if (roi.groups.length) {
      roi.groups = [];
      roi.selectedGroup = null;
      updateSettingsBinding();
    }
    panel.classList.add("hidden");
    updateRoiControls();
    return;
  }
  panel.classList.remove("hidden");
  if (roi.groupEditing) panel.open = true;
  $("roi-region-count").textContent = roi.regions.length + (roi.current.length ? " +1 进行中" : "");
  $("roi-point-count").textContent = roi.regions.reduce((n, r) => n + r.points.length, 0) + roi.current.length;
  $("roi-panel-actions").classList.toggle("hidden", !roi.groupEditing);

  const list = $("roi-region-list");
  list.innerHTML = "";
  const fmt = (p) => `${p.lon != null ? p.lon.toFixed(6) : "-"}, ${p.lat != null ? p.lat.toFixed(6) : "-"}`;

  /* 区域行（编辑分组模式时前面带勾选框；已入组的区域灰勾禁选） */
  const groupedRegions = new Set();
  roi.groups.forEach((g) => g.regions.forEach((i) => groupedRegions.add(i)));

  roi.regions.forEach((region, idx) => {
    const div = document.createElement("div");
    div.className = "roi-region";
    const inGroup = groupedRegions.has(idx);
    const picked = roi.groupPick.has(idx) && !inGroup;
    const cb = roi.groupEditing
      ? `<input type="checkbox" class="rg-pick" data-region="${idx}"` +
        `${picked ? " checked" : ""}${inGroup ? " checked disabled" : ""}` +
        ` title="${inGroup ? "该区域已加入分组，不能重复添加" : "勾选后可添加为分组"}">`
      : "";
    const ptsText = region.points.map((p, i) => `${i + 1}.(${fmt(p)})`).join(" ");
    div.innerHTML = `
      ${cb}
      <span class="rr-title">区域${idx + 1}</span>
      <span class="rr-pts">凸包顶点 ${region.hull.length} / 标点 ${region.points.length}：<br>${ptsText}</span>
      <button class="rr-del" title="删除该区域">×</button>`;
    div.querySelector(".rr-del").addEventListener("click", () => {
      roi.regions.splice(idx, 1);
      roi.groupPick.delete(idx);
      remapGroupsAfterRegionDelete(idx);
      drawRoi();
      updateRoiPanel();
      refreshRoiArea();
    });
    if (roi.groupEditing) {
      div.querySelector(".rg-pick").addEventListener("change", (e) => {
        const pickedIdx = parseInt(e.target.dataset.region, 10);
        if (e.target.checked) roi.groupPick.add(pickedIdx); else roi.groupPick.delete(pickedIdx);
      });
    }
    list.appendChild(div);
  });

  if (roi.current.length) {
    const div = document.createElement("div");
    div.className = "roi-region";
    const ptsText = roi.current.map((p, i) => `${i + 1}.(${fmt(p)})`).join(" ");
    div.innerHTML = `
      <span class="rr-title">进行中</span>
      <span class="rr-pts">已标 ${roi.current.length} 个点（右键完成）：<br>${ptsText}</span>
      <button class="rr-del" title="取消当前区域">×</button>`;
    div.querySelector(".rr-del").addEventListener("click", () => {
      roi.current = [];
      drawRoi();
      updateRoiPanel();
    });
    list.appendChild(div);
  }

  /* 分组行：点击在「诊断分级与施肥映射设置」中显示/编辑该分组；未分组区域用全局设置 */
  if (roi.groups.length) {
    const head = document.createElement("div");
    head.className = "roi-groups-head";
    head.textContent = "分组（点击选择后可在「诊断分级与施肥映射设置」中编辑；未分组区域使用全局设置）";
    list.appendChild(head);

    const globalRow = document.createElement("div");
    globalRow.className = "roi-region roi-group-row" + (roi.selectedGroup == null ? " selected" : "");
    globalRow.innerHTML = `
      <span class="rr-title">🌐 全局设置</span>
      <span class="rr-pts">未加入分组的区域使用这里的分级与施肥映射</span>`;
    globalRow.addEventListener("click", () => selectGroup(null));
    list.appendChild(globalRow);

    roi.groups.forEach((g, gi) => {
      const row = document.createElement("div");
      row.className = "roi-region roi-group-row" + (roi.selectedGroup === gi ? " selected" : "");
      row.innerHTML = `
        <input type="text" class="rr-name" value="${escapeHtml(g.name || `组${gi + 1}`)}" title="可重命名分组名称">
        <span class="rr-pts">${g.regions.length ? "含 " + g.regions.map((i) => "区域" + (i + 1)).join("、") : "未勾选区域"}</span>
        <button class="rr-del" title="删除该分组（其区域回到未分组）">×</button>`;
      row.addEventListener("click", (e) => {
        if (e.target.closest(".rr-del") || e.target.closest(".rr-name")) return;
        selectGroup(gi);
      });
      row.querySelector(".rr-name").addEventListener("click", (e) => e.stopPropagation());
      row.querySelector(".rr-name").addEventListener("input", (e) => {
        g.name = e.target.value.trim();
      });
      row.querySelector(".rr-del").addEventListener("click", (e) => {
        e.stopPropagation();
        roi.groups.splice(gi, 1);
        if (roi.selectedGroup === gi) {
          roi.selectedGroup = null;
          applySettingsToControls(roi.globalSnapshot || captureControlSettings());
        } else if (roi.selectedGroup != null && roi.selectedGroup > gi) {
          roi.selectedGroup -= 1;
        }
        updateSettingsBinding();
        updateRoiPanel();
      });
      list.appendChild(row);
    });
  }
  updateRoiControls();
  updateSettingsBinding();
  syncLeftColumnHeight();
}

/* ---------------- ROI 区域分组 ---------------- */

/* 把勾选的区域组成一个新分组（区域若已属于其他分组则移入新分组），
   并选中新分组，使其分级与施肥映射出现在主表格中 */
function addGroupFromRegions() {
  const picked = Array.from(roi.groupPick).sort((a, b) => a - b);
  if (!picked.length) {
    toast("请先勾选要加入分组的区域", true);
    return;
  }
  roi.groups.forEach((other) => {
    other.regions = other.regions.filter((i) => !picked.includes(i));
  });
  roi.groups.push({
    regions: picked,
    level_count: parseInt($("level-count").value, 10) || 5,
    mode: document.querySelector('input[name="mapping-mode"]:checked').value,
    formula: { b: $("f-base-level").value, qb: $("f-base-fert").value, k: $("f-coeff").value },
    manual: {},
    quantiles: readLevelTable().map((r) => r.quantile),
    labels: readLevelTable().map((r) => r.label),
  });
  const gi = roi.groups.length - 1;
  readLevelTable().forEach((r) => { roi.groups[gi].manual[r.level] = r.fertilizer; });
  roi.groupPick.clear();
  updateRoiPanel();
  selectGroup(gi);
  toast(`分组「组${gi + 1}」已添加（区域 ${picked.map((i) => i + 1).join("、")}），可继续勾选其他区域添加分组`);
}

function finishGroupEditing() {
  roi.groupEditing = false;
  roi.groupPick.clear();
  updateRoiControls();
  updateRoiPanel();
  toast("已退出分组编辑");
}

function initViewer() {
  const stage = $("viewer-stage");

  $("viewer-tabs").addEventListener("click", (e) => {    const btn = e.target.closest(".tab");
    if (!btn || btn.disabled) return;
    showPreview(btn.dataset.view);
  });

  stage.addEventListener("wheel", (e) => {
    if (!viewer.naturalW) return;
    e.preventDefault();
    const rect = stage.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newScale = Math.min(Math.max(viewer.scale * factor, 0.02), 60);
    viewer.tx = mx - (mx - viewer.tx) * (newScale / viewer.scale);
    viewer.ty = my - (my - viewer.ty) * (newScale / viewer.scale);
    viewer.scale = newScale;
    applyViewerTransform();
  }, { passive: false });

  let dragging = false, lastX = 0, lastY = 0, downX = 0, downY = 0, movedFar = false;
  stage.addEventListener("pointerdown", (e) => {
    if (!viewer.naturalW || e.button !== 0) return;
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    downX = e.clientX; downY = e.clientY; movedFar = false;
    stage.classList.add("dragging");
    try { stage.setPointerCapture(e.pointerId); } catch (err) { /* 合成事件无活动指针 */ }
  });
  stage.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    if (Math.abs(e.clientX - downX) + Math.abs(e.clientY - downY) > 4) movedFar = true;
    viewer.tx += e.clientX - lastX;
    viewer.ty += e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    applyViewerTransform();
  });
  stage.addEventListener("pointerup", (e) => {
    dragging = false;
    stage.classList.remove("dragging");
    if (roi.editing && !movedFar && state.currentView === "input" && e.button === 0) {
      const rect = stage.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      addRoiPoint((mx - viewer.tx) / viewer.scale, (my - viewer.ty) / viewer.scale);
    }
  });
  stage.addEventListener("contextmenu", (e) => {
    if (!roi.editing) return;
    e.preventDefault();
    finalizeCurrentRegion();
  });

  $("btn-roi-edit").addEventListener("click", () => setRoiEditing(!roi.editing));
  $("btn-roi-done").addEventListener("click", () => setRoiEditing(false));
  $("btn-roi-undo").addEventListener("click", undoRoiPoint);
  $("btn-roi-clear-current").addEventListener("click", () => {
    if (!roi.current.length) { toast("当前没有进行中的区域"); return; }
    roi.current = [];
    drawRoi();
    updateRoiPanel();
  });
  $("btn-roi-clear-all").addEventListener("click", () => {
    if (!roi.regions.length && !roi.current.length) return;
    roi.regions = [];
    roi.current = [];
    roi.groups = [];
    drawRoi();
    updateRoiPanel();
    refreshRoiArea();
    toast("已清空全部感兴趣区域");
  });

  /* 区域分组：在感兴趣区域列表中勾选区域 → 添加为分组 → 完成 */
  document.querySelectorAll('input[name="make-mode"]').forEach((radio) => {
    radio.addEventListener("change", () => { updateRoiControls(); });
  });
  $("btn-roi-groups").addEventListener("click", () => {
    document.querySelector('input[name="make-mode"][value="roi"]').checked = true;
    updateRoiControls();
    roi.groupEditing = !roi.groupEditing;
    roi.groupPick.clear();
    if (roi.groupEditing) {
      $("roi-panel").open = true;
      updateRoiPanel();
      $("roi-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
    } else {
      updateRoiPanel();
    }
  });
  $("btn-group-add").addEventListener("click", addGroupFromRegions);
  $("btn-group-finish").addEventListener("click", finishGroupEditing);

  $("btn-zoom-in").addEventListener("click", () => { viewer.scale = Math.min(viewer.scale * 1.25, 60); applyViewerTransform(); });
  $("btn-zoom-out").addEventListener("click", () => { viewer.scale = Math.max(viewer.scale / 1.25, 0.02); applyViewerTransform(); });
  $("btn-fit").addEventListener("click", fitViewer);
  window.addEventListener("resize", () => {
    fitViewer();
    syncLeftColumnHeight();
  });
  $("roi-panel").addEventListener("toggle", syncLeftColumnHeight);
}

/* ---------------- dj2pr 模板替换 ---------------- */
function initDj2pr() {
  $("btn-dj2pr").addEventListener("click", async () => {
    const templatePath = $("dj2pr-template").value.trim();
    if (!templatePath) { toast("请填写模板 tif 路径", true); return; }
    try {
      $("card-progress").classList.remove("hidden");
      await api("/api/dj2pr", { template_path: templatePath, method: $("dj2pr-method").value });
      state.running = true;
      pollJob();
    } catch (err) { toast(err.message, true); }
  });

  $("btn-open-folder").addEventListener("click", async () => {
    const dir = state.result && state.result.output_dir;
    if (!dir) { toast("尚无生成结果", true); return; }
    try {
      await api("/api/reveal", { path: dir });
    } catch (err) { toast(err.message, true); }
  });
}

function renderDj2prReport(report) {
  $("dj2pr-report").innerHTML = `
    替换完成：<b>${report.new_fertilizer_path}</b><br>
    模板网格 ${report.template_size[0]} × ${report.template_size[1]}（${report.template_dtype}），
    重采样方式 ${report.resampling}，替换有效像元 ${report.replaced_pixel_count.toLocaleString()} 个。<br>
    可在“打开输出文件夹”外的路径直接取用该文件导入无人机。
  `;
  toast("模板替换完成");
}

/* ---------------- 初始化 ---------------- */
async function init() {
  try {
    state.config = await api("/api/config");
  } catch (err) {
    toast("无法加载配置: " + err.message, true);
    return;
  }
  initInput();
  initParams();
  initViewer();
  initDj2pr();
  await refreshSession();
}

init();
