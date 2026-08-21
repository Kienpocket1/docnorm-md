const $ = (selector) => document.querySelector(selector);
const state = { file: null, jobId: null, originalMarkdown: "", pollTimer: null };
const stageOrder = ["UPLOAD", "EXTRACT", "VERIFY", "NORMALIZE", "DONE"];
const maximumUploadBytes = Number(document.body.dataset.maxUploadBytes || 104857600);

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.remove("is-hidden");
  window.clearTimeout(node._timer);
  node._timer = window.setTimeout(() => node.classList.add("is-hidden"), 4200);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `Lỗi HTTP ${response.status}`;
    try { detail = (await response.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return response.json();
}

function selectFile(file) {
  if (!file) return;
  const extension = file.name.toLowerCase().split(".").pop();
  if (!["pdf", "docx"].includes(extension)) return toast("Chỉ hỗ trợ tệp PDF hoặc DOCX.");
  if (file.size > maximumUploadBytes) return toast("Tệp vượt quá giới hạn dung lượng.");
  state.file = file;
  $("#fileName").textContent = file.name;
  $("#fileInfo").textContent = `${formatBytes(file.size)} · ${extension.toUpperCase()}`;
  $("#selectedFile").classList.remove("is-hidden");
  $("#dropzone").classList.add("is-hidden");
  $("#startButton").disabled = false;
}

function clearFile() {
  state.file = null;
  $("#fileInput").value = "";
  $("#selectedFile").classList.add("is-hidden");
  $("#dropzone").classList.remove("is-hidden");
  $("#startButton").disabled = true;
}

function updateProgress(job) {
  $("#progressPanel").classList.remove("is-hidden");
  $("#progressPercent").textContent = `${job.progress_percent}%`;
  $("#progressBar").style.width = `${job.progress_percent}%`;
  $("#progressMessage").textContent = job.message;
  $("#elapsedTime").textContent = `${Number(job.elapsed_seconds || 0).toFixed(1)} giây`;
  const current = Math.max(0, stageOrder.indexOf(job.stage));
  document.querySelectorAll("#stageList li").forEach((node, index) => {
    node.classList.toggle("active", index === current);
    node.classList.toggle("done", index < current || job.status === "COMPLETED");
  });
}

async function pollJob(jobId) {
  window.clearTimeout(state.pollTimer);
  try {
    const job = await api(`/api/jobs/${jobId}`);
    updateProgress(job);
    if (job.status === "COMPLETED") {
      await showResult(jobId);
      loadHistory();
      return;
    }
    if (job.status === "FAILED") {
      $("#startButton").disabled = false;
      $("#progressMessage").textContent = job.message;
      $("#progressPanel .spinner").classList.add("is-hidden");
      toast(`${job.error_code || "CONVERSION_FAILED"}: ${job.message}`);
      loadHistory();
      return;
    }
    state.pollTimer = window.setTimeout(() => pollJob(jobId), 1000);
  } catch {
    state.pollTimer = window.setTimeout(() => pollJob(jobId), 1800);
  }
}

async function showResult(jobId) {
  state.jobId = jobId;
  const result = await api(`/api/jobs/${jobId}/result`);
  const metrics = result.metrics;
  state.originalMarkdown = result.markdown;
  $("#markdownEditor").value = result.markdown;
  $("#markdownPreview").textContent = result.markdown;
  $("#editorMeta").textContent = `${result.markdown.split("\n").length} dòng · ${result.markdown.length.toLocaleString("vi-VN")} ký tự`;
  $("#saveMarkdown").disabled = true;
  $("#dirtyDot").classList.add("is-hidden");
  $("#metricPages").textContent = metrics.pages ?? "Không xác định";
  $("#metricElements").textContent = metrics.elements;
  $("#metricTables").textContent = metrics.tables;
  $("#metricImages").textContent = metrics.images;
  $("#metricCoverage").textContent = `${Math.round(metrics.provenance_coverage * 100)}%`;
  $("#metricBroken").textContent = metrics.broken_characters;
  $("#metricVlmPages").textContent = metrics.vlm_pages || 0;
  const disposition = result.quality.disposition;
  const badge = $("#qualityBadge");
  badge.textContent = disposition;
  badge.className = `quality-badge ${disposition === "PASS" ? "" : disposition === "NEEDS_REVIEW" ? "error" : "warning"}`;
  const coverage = Math.round(metrics.provenance_coverage * 100);
  $("#coverageLabel").textContent = `${coverage}%`;
  $("#coverageBar").style.width = `${coverage}%`;
  $("#engineChain").textContent = result.quality.engine_chain.join(" → ") || "—";
  const repairProfile = result.quality.repair_profile_applied || "generic";
  const repairVersion = result.quality.repair_version ? ` · ${result.quality.repair_version}` : "";
  $("#repairProfileApplied").textContent = `${repairProfile.toUpperCase()}${repairVersion}`;
  renderIssues(result.quality.issues);
  $("#downloadMarkdown").href = result.downloads.markdown;
  $("#downloadBundle").href = result.downloads.bundle;
  $("#downloadReport").href = result.downloads.report;
  $("#resultPanel").classList.remove("is-hidden");
  $("#progressPanel .spinner").classList.add("is-hidden");
  $("#resultPanel").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderIssues(issues) {
  const list = $("#issueList");
  list.replaceChildren();
  $("#issueCount").textContent = `${issues.length} vấn đề`;
  if (!issues.length) {
    const item = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = "Không phát hiện lỗi cấu trúc";
    item.append(title);
    list.append(item);
    return;
  }
  issues.forEach((issue) => {
    const item = document.createElement("li");
    item.className = issue.severity.toLowerCase();
    const title = document.createElement("strong");
    title.textContent = `${issue.severity} · ${issue.code}`;
    const body = document.createElement("span");
    body.textContent = issue.message;
    item.append(title, body);
    list.append(item);
  });
}

async function loadHistory() {
  try {
    const payload = await api("/api/jobs?limit=8");
    const root = $("#historyList");
    root.replaceChildren();
    if (!payload.items.length) {
      const empty = document.createElement("p");
      empty.className = "empty-state";
      empty.textContent = "Chưa có tài liệu nào được xử lý.";
      root.append(empty);
      return;
    }
    payload.items.forEach((job) => {
      const row = document.createElement("article");
      row.className = "history-item";
      const name = document.createElement("div");
      name.className = "history-name";
      const strong = document.createElement("strong"); strong.textContent = job.filename;
      const small = document.createElement("small"); small.textContent = new Date(job.created_at).toLocaleString("vi-VN");
      name.append(strong, small);
      const mode = document.createElement("span"); mode.className = "history-meta"; mode.textContent = job.mode.toUpperCase();
      const status = document.createElement("span"); status.className = "history-status"; status.textContent = job.status;
      const action = document.createElement("button"); action.className = "button secondary"; action.type = "button";
      action.textContent = job.status === "COMPLETED" ? "Mở" : "Xem";
      action.addEventListener("click", () => job.status === "COMPLETED" ? showResult(job.job_id) : pollJob(job.job_id));
      const remove = document.createElement("button"); remove.className = "button ghost"; remove.type = "button"; remove.textContent = "Xóa";
      remove.addEventListener("click", async () => {
        if (!window.confirm(`Xóa kết quả ${job.filename}?`)) return;
        try {
          const response = await fetch(`/api/jobs/${job.job_id}`, { method: "DELETE" });
          if (!response.ok) throw new Error("Không thể xóa job này.");
          if (state.jobId === job.job_id) $("#resultPanel").classList.add("is-hidden");
          loadHistory();
          toast("Đã xóa dữ liệu job khỏi máy.");
        } catch (error) { toast(error.message); }
      });
      row.append(name, mode, status, action, remove);
      root.append(row);
    });
  } catch {}
}

$("#fileInput").addEventListener("change", (event) => selectFile(event.target.files[0]));
$("#removeFile").addEventListener("click", clearFile);
const dropzone = $("#dropzone");
["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.remove("dragging"); }));
dropzone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
document.querySelectorAll(".mode-card input").forEach((input) => input.addEventListener("change", () => {
  document.querySelectorAll(".mode-card").forEach((card) => card.classList.toggle("selected", card.querySelector("input").checked));
}));

$("#uploadForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.file) return;
  const form = new FormData();
  form.append("file", state.file);
  form.append("mode", document.querySelector("input[name=mode]:checked").value);
  form.append("repair_profile", $("#repairProfile").value);
  $("#startButton").disabled = true;
  $("#resultPanel").classList.add("is-hidden");
  $("#progressPanel .spinner").classList.remove("is-hidden");
  updateProgress({ progress_percent: 5, message: "Đang tải tài liệu lên bộ xử lý cục bộ", stage: "UPLOAD", elapsed_seconds: 0, status: "QUEUED" });
  try {
    const job = await api("/api/jobs", { method: "POST", body: form });
    state.jobId = job.job_id;
    pollJob(job.job_id);
  } catch (error) {
    $("#startButton").disabled = false;
    toast(error.message);
  }
});

$("#markdownEditor").addEventListener("input", (event) => {
  const dirty = event.target.value !== state.originalMarkdown;
  $("#saveMarkdown").disabled = !dirty;
  $("#dirtyDot").classList.toggle("is-hidden", !dirty);
  $("#editorMeta").textContent = `${event.target.value.split("\n").length} dòng · ${event.target.value.length.toLocaleString("vi-VN")} ký tự`;
});
$("#saveMarkdown").addEventListener("click", async () => {
  try {
    await api(`/api/jobs/${state.jobId}/markdown`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: $("#markdownEditor").value }),
    });
    state.originalMarkdown = $("#markdownEditor").value;
    $("#markdownPreview").textContent = state.originalMarkdown;
    $("#saveMarkdown").disabled = true;
    $("#dirtyDot").classList.add("is-hidden");
    toast("Đã lưu Markdown và tạo lại gói kết quả.");
  } catch (error) { toast(error.message); }
});
$("#sourceTab").addEventListener("click", () => {
  $("#sourceTab").classList.add("active"); $("#previewTab").classList.remove("active");
  $("#markdownEditor").classList.remove("is-hidden"); $("#markdownPreview").classList.add("is-hidden");
});
$("#previewTab").addEventListener("click", () => {
  $("#markdownPreview").textContent = $("#markdownEditor").value;
  $("#previewTab").classList.add("active"); $("#sourceTab").classList.remove("active");
  $("#markdownPreview").classList.remove("is-hidden"); $("#markdownEditor").classList.add("is-hidden");
});
$("#refreshHistory").addEventListener("click", loadHistory);

api("/api/health").then((health) => {
  const status = $("#engineStatus");
  if (health.capabilities.geometry_ocr_installed && health.capabilities.math_vlm_installed) status.textContent = "Geometry OCR + Math VLM sẵn sàng";
  else if (health.capabilities.geometry_ocr_installed) status.textContent = "Geometry OCR GPU sẵn sàng";
  else if (health.capabilities.hybrid_available && health.capabilities.math_vlm_installed) status.textContent = "Hybrid OCR + Math VLM sẵn sàng";
  else if (health.capabilities.hybrid_available) status.textContent = "Pipeline + Hybrid OCR sẵn sàng";
  else if (health.capabilities.rag_pipeline_available) status.textContent = "Pipeline sẵn sàng · OCR chưa chạy";
  else status.textContent = "Chế độ fallback sẵn sàng";
}).catch(() => { $("#engineStatus").textContent = "Không kết nối được backend"; });
loadHistory();
