const key = "news-assistant-feedback-v2";
let storageAvailable = true;

function readEvents() {
  try {
    const value = JSON.parse(localStorage.getItem(key) || "{}");
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  } catch (_) { storageAvailable = false; return {}; }
}
function writeEvents(events) {
  try {
    localStorage.setItem(key, JSON.stringify(events));
    storageAvailable = true;
    return true;
  } catch (_) { storageAvailable = false; return false; }
}
function renderFeedback(card, entry, message) {
  card.querySelectorAll("[data-feedback]").forEach(button => {
    const action = button.dataset.feedback;
    const selected = action === "track" ? Boolean(entry?.tracking) : entry?.vote === action;
    button.setAttribute("aria-pressed", String(selected));
    button.classList.toggle("selected", selected);
  });
  const reason = card.querySelector(".feedback-reason");
  if (reason) {
    reason.hidden = entry?.vote !== "reduce";
    reason.value = entry?.reason || "irrelevant";
  }
  const status = card.querySelector(".feedback-status");
  status.hidden = false;
  status.textContent = message || (entry?.vote || entry?.tracking
    ? "已保存在此浏览器；导出并导入后用于推荐评估"
    : "已撤销；再次导出可同步撤销");
  updateExport();
}
function setFeedback(card, action) {
  const data = readEvents();
  const ref = card.dataset.eventRef;
  if (!ref) return;
  const previous = data[ref] || { event_ref: ref, vote: null, reason: null, tracking: false };
  const timestamp = Math.max(Date.now(), (Date.parse(previous.updated_at) || 0) + 1);
  const entry = { ...previous, updated_at: new Date(timestamp).toISOString() };
  if (action === "track") entry.tracking = !entry.tracking;
  else if (action === "reason") entry.reason = card.querySelector(".feedback-reason").value;
  else {
    entry.vote = entry.vote === action ? null : action;
    entry.reason = entry.vote === "reduce" ? (card.querySelector(".feedback-reason")?.value || "irrelevant") : null;
  }
  // Preserve undo records so importing again also revokes previous feedback.
  data[ref] = entry;
  if (writeEvents(data)) renderFeedback(card, entry);
  else renderFeedback(card, previous, "浏览器不允许保存反馈");
}
function validEvents() {
  return Object.values(readEvents()).filter(entry => entry &&
    typeof entry.event_ref === "string" && /^[a-f0-9]{24}$/.test(entry.event_ref) &&
    ["helpful", "reduce", null].includes(entry.vote) &&
    typeof entry.tracking === "boolean" && typeof entry.updated_at === "string");
}
function updateExport() {
  const button = document.querySelector("#export");
  const status = document.querySelector("#export-status");
  if (!button || !status) return;
  const count = validEvents().length;
  button.disabled = !storageAvailable || count === 0;
  status.textContent = !storageAvailable ? "浏览器不允许保存反馈"
    : count ? `已有 ${count} 条反馈记录可导出（含撤销）` : "暂无可导出的反馈";
}
function exportPreferences() {
  const entries = validEvents();
  if (!entries.length) return;
  const payload = { schema_version: "news-feedback-v2", events: entries };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "news-feedback.json";
  link.click();
  URL.revokeObjectURL(link.href);
}
document.querySelectorAll("article").forEach(card => {
  const ref = card.dataset.eventRef;
  if (!ref) {
    card.querySelectorAll("[data-feedback]").forEach(button => { button.disabled = true; });
    renderFeedback(card, null, "此历史页面仅供阅读；请在新版日报中提交反馈");
    return;
  }
  const entry = readEvents()[ref];
  if (entry) renderFeedback(card, entry);
  card.querySelectorAll("[data-feedback]").forEach(button =>
    button.addEventListener("click", () => setFeedback(card, button.dataset.feedback)));
  card.querySelector(".feedback-reason")?.addEventListener("change", () => setFeedback(card, "reason"));
});
document.querySelector("#export")?.addEventListener("click", exportPreferences);
updateExport();
if (location.hash) {
  try {
    const target = document.getElementById(location.hash.slice(1)) || document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (target) {
      target.scrollIntoView({ block: "center" });
      target.classList.add("target-highlight");
      setTimeout(() => target.classList.remove("target-highlight"), 1800);
    }
  } catch (_) { /* Invalid fragments must not disable feedback. */ }
}
