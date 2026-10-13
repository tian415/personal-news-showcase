// Explicit reading-value ratings. Browser fallback plus same-origin private persistence.
(() => {
  'use strict';
  const root = document.querySelector('#draft-feedback');
  const status = root.querySelector('[role="status"]');
  const requestHash = root.dataset.requestHash, draftHash = root.dataset.draftHash;
  const key = 'daily-writer-feedback-v2:' + requestHash + ':' + draftHash;
  const readButtons = [...document.querySelectorAll('[data-action="read"]')];
  const radios = [...document.querySelectorAll('.rating input[type="radio"]')];
  const allowed = new Set(readButtons.map(b => b.dataset.item));
  const reasons = new Set([null, 'unclear', 'repeated', 'not_relevant', 'unreliable', 'insightful']);
  let rows = [], pending = false, diskSaved = false, syncFailed = false, revision = 0;
  let queue = Promise.resolve();
  function valid(saved) {
    return saved.schema === 'daily-writer-draft-feedback-v2' && saved.request_hash === requestHash &&
      saved.draft_hash === draftHash && saved.rating_dimension === 'reading_value' &&
      Array.isArray(saved.items) && saved.items.every(row => allowed.has(row.item_id) &&
        typeof row.read === 'boolean' && reasons.has(row.reason) &&
        (row.rating === null || (Number.isInteger(row.rating) && row.rating >= 1 && row.rating <= 5))) &&
      new Set(saved.items.map(row => row.item_id)).size === saved.items.length;
  }
  try {
    const raw = localStorage.getItem(key);
    if (raw) {
      const saved = JSON.parse(raw);
      if (valid(saved)) { rows = saved.items; pending = saved.pending === true; }
    }
  } catch (_) { /* Preserve unknown storage; server or export remains available. */ }
  function envelope() {
    return {schema: 'daily-writer-draft-feedback-v2', request_hash: requestHash,
      draft_hash: draftHash, scope: 'local_draft_only', approval: false,
      rating_dimension: 'reading_value', scale: {min: 1, max: 5}, items: rows};
  }
  function cache() {
    try { localStorage.setItem(key, JSON.stringify({...envelope(), pending})); } catch (_) { /* Export remains available. */ }
  }
  const rowFor = item => rows.find(row => row.item_id === item);
  function refresh() {
    readButtons.forEach(button => button.setAttribute('aria-pressed', String(rowFor(button.dataset.item)?.read === true)));
    radios.forEach(radio => {
      const score = rowFor(radio.dataset.item)?.rating ?? null;
      radio.checked = score === Number(radio.value);
      radio.closest('.star-option').classList.toggle('is-filled', score !== null && Number(radio.value) <= score);
    });
    document.querySelectorAll('[data-rating-status]').forEach(label => {
      const score = rowFor(label.dataset.ratingStatus)?.rating;
      label.textContent = score ? `已评 ${score} / 5 星 · 阅读价值，可随时修改` : '未评分 · 1星收获很少，5星很值得读';
    });
    document.querySelectorAll('[data-reason]').forEach(select => { select.value = rowFor(select.dataset.reason)?.reason ?? ''; });
    const count = rows.filter(row => row.rating !== null).length;
    status.textContent = syncFailed ? `本地项目暂未保存成功；${count} 篇评分可导出备份，稍后操作会再次尝试保存。`
      : pending ? '正在保存到本地项目…'
      : diskSaved ? `已保存到本地项目：${count} 篇评分。后续本地试稿可读取。` : '正在读取本地反馈…';
  }
  function sync() {
    const snapshot = envelope(), current = revision;
    queue = queue.then(async () => {
      try {
        const reply = await fetch('/feedback', {method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(snapshot), signal: AbortSignal.timeout(5000)});
        if (!reply.ok || !(await reply.json()).saved) throw new Error('save');
        if (revision === current) { pending = false; diskSaved = true; syncFailed = false; cache(); }
      } catch (_) { if (revision === current) syncFailed = true; }
      refresh();
    });
  }
  function update(item, changes) {
    const row = {...(rowFor(item) ?? {item_id: item, read: false, rating: null, reason: null}),
      ...changes, updated_at: new Date().toISOString()};
    rows = rows.filter(old => old.item_id !== item);
    if (row.read || row.rating !== null || row.reason !== null) rows.push(row);
    revision++; pending = true; syncFailed = false; cache(); refresh(); sync();
  }
  readButtons.forEach(button => button.addEventListener('click', () => update(button.dataset.item, {read: !rowFor(button.dataset.item)?.read})));
  radios.forEach(radio => radio.addEventListener('change', () => update(radio.dataset.item, {rating: Number(radio.value)})));
  document.querySelectorAll('[data-clear]').forEach(button => button.addEventListener('click', () => update(button.dataset.clear, {rating: null})));
  document.querySelectorAll('[data-reason]').forEach(select => select.addEventListener('change', () => update(select.dataset.reason, {reason: select.value || null})));
  root.querySelector('button').addEventListener('click', () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(envelope(), null, 2)], {type: 'application/json'}));
    const link = document.createElement('a'); link.href = url;
    link.download = 'draft-feedback-' + draftHash.slice(0, 12) + '.json'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  refresh();
  (async () => {
    try {
      const reply = await fetch('/feedback', {signal: AbortSignal.timeout(5000)});
      if (!reply.ok) throw new Error('read');
      const saved = await reply.json();
      if (!valid(saved)) throw new Error('binding');
      if (revision === 0) {
        if (pending) sync();
        else { rows = saved.items; diskSaved = true; cache(); }
      }
    } catch (_) { syncFailed = true; }
    refresh();
  })();
})();
