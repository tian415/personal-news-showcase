import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const SCRIPT_PATH = new URL("../src/news_assistant/static/feedback.js", import.meta.url);
const SCRIPT = await readFile(SCRIPT_PATH, "utf8");

class FakeClassList {
  constructor() {
    this.values = new Set();
  }

  toggle(name, enabled) {
    if (enabled) this.values.add(name);
    else this.values.delete(name);
  }
  add(name) { this.values.add(name); }
  remove(name) { this.values.delete(name); }
}

class FakeElement {
  constructor({ dataset = {}, selector = "" } = {}) {
    this.dataset = { ...dataset };
    this.selector = selector;
    this.children = [];
    this.listeners = new Map();
    this.attributes = new Map();
    this.classList = new FakeClassList();
    this.hidden = false;
    this.disabled = false;
    this.value = "irrelevant";
    this.textContent = "";
  }

  querySelectorAll(selector) {
    return this.children.filter((child) =>
      selector === "[data-feedback]" ? child.dataset.feedback : false,
    );
  }

  querySelector(selector) {
    return this.children.find((child) => child.selector === selector) ?? null;
  }

  addEventListener(type, handler) {
    this.listeners.set(type, handler);
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  click() {
    this.listeners.get("click")?.();
  }

  change() {
    this.listeners.get("change")?.();
  }
}

class FakeDocument {
  constructor(card) {
    this.card = card;
    this.exportButton = new FakeElement({ selector: "#export" });
    this.exportStatus = new FakeElement({ selector: "#export-status" });
    this.createdLinks = [];
  }

  querySelectorAll(selector) {
    return selector === "article" ? [this.card] : [];
  }

  querySelector(selector) {
    if (selector === "#export") return this.exportButton;
    if (selector === "#export-status") return this.exportStatus;
    return null;
  }

  createElement(name) {
    assert.equal(name, "a");
    const link = new FakeElement();
    this.createdLinks.push(link);
    return link;
  }

  getElementById(id) {
    return id === this.card.id ? this.card : null;
  }
}

class FakeStorage {
  constructor({ failWrites = false } = {}) {
    this.values = new Map();
    this.failWrites = failWrites;
  }

  getItem(name) {
    return this.values.get(name) ?? null;
  }

  setItem(name, value) {
    if (this.failWrites) throw new Error("storage disabled");
    this.values.set(name, value);
  }
}

class FakeBlob {
  constructor(parts, options) {
    this.textValue = parts.join("");
    this.type = options.type;
  }
}

function makeCard(eventRef = "0123456789abcdef01234567") {
  const card = new FakeElement({ dataset: { eventRef }, selector: "article" });
  card.children.push(
    new FakeElement({ dataset: { feedback: "helpful" } }),
    new FakeElement({ dataset: { feedback: "reduce" } }),
    new FakeElement({ dataset: { feedback: "track" } }),
    new FakeElement({ selector: ".feedback-reason" }),
    new FakeElement({ selector: ".feedback-status" }),
  );
  return card;
}

function runFeedback({ storage = new FakeStorage(), hash = "", eventRef } = {}) {
  const card = makeCard(eventRef);
  card.id = "article-quality%3A2";
  card.scrollIntoView = () => { card.scrolled = true; };
  const document = new FakeDocument(card);
  const blobs = [];
  const revoked = [];
  const context = {
    Blob: class extends FakeBlob {
      constructor(parts, options) {
        super(parts, options);
        blobs.push(this);
      }
    },
    URL: {
      createObjectURL: () => "blob:test",
      revokeObjectURL: (url) => revoked.push(url),
    },
    document,
    localStorage: storage,
    location: { hash },
    setTimeout: () => {},
  };
  vm.runInNewContext(SCRIPT, context, { filename: "feedback.js" });
  return { card, document, storage, blobs, revoked };
}

function storedEvents(storage) {
  return JSON.parse(storage.getItem("news-assistant-feedback-v2"));
}

test("registered vote and tracking handlers persist independently", () => {
  const state = runFeedback();
  const [helpful, , track] = state.card.querySelectorAll("[data-feedback]");

  helpful.click();
  track.click();

  assert.equal(storedEvents(state.storage)["0123456789abcdef01234567"].vote, "helpful");
  assert.equal(storedEvents(state.storage)["0123456789abcdef01234567"].tracking, true);
  assert.equal(helpful.attributes.get("aria-pressed"), "true");
  assert.equal(track.attributes.get("aria-pressed"), "true");
});

test("clicking a selected vote creates a revoke tombstone", () => {
  const state = runFeedback();
  const [helpful] = state.card.querySelectorAll("[data-feedback]");

  helpful.click();
  helpful.click();

  const entry = storedEvents(state.storage)["0123456789abcdef01234567"];
  assert.equal(entry.vote, null);
  assert.equal(entry.tracking, false);
  assert.equal(typeof entry.updated_at, "string");
  assert.equal(state.document.exportButton.disabled, false);
});

test("export handler emits schema and tombstones", () => {
  const state = runFeedback();
  const [helpful] = state.card.querySelectorAll("[data-feedback]");

  helpful.click();
  helpful.click();
  state.document.exportButton.click();

  assert.equal(state.blobs.length, 1);
  const payload = JSON.parse(state.blobs[0].textValue);
  assert.equal(payload.schema_version, "news-feedback-v2");
  assert.equal(payload.events.length, 1);
  assert.equal(payload.events[0].event_ref, "0123456789abcdef01234567");
  assert.equal(payload.events[0].vote, null);
  assert.deepEqual(state.revoked, ["blob:test"]);
});

test("storage failure is visible and disables export", () => {
  const state = runFeedback({ storage: new FakeStorage({ failWrites: true }) });
  const [helpful] = state.card.querySelectorAll("[data-feedback]");

  helpful.click();

  assert.equal(state.storage.getItem("news-assistant-feedback-v2"), null);
  assert.equal(state.document.exportButton.disabled, true);
  assert.match(state.card.querySelector(".feedback-status").textContent, /不允许保存反馈/);
});

test("encoded article link scrolls to the exact DOM id", () => {
  const state = runFeedback({ hash: "#article-quality%3A2" });
  assert.equal(state.card.scrolled, true);
  assert.equal(state.card.classList.values.has("target-highlight"), true);
});

test("failed write preserves the saved vote and refresh restores it", () => {
  const storage = new FakeStorage();
  const first = runFeedback({ storage });
  first.card.querySelectorAll("[data-feedback]")[0].click();
  storage.failWrites = true;
  const page = runFeedback({ storage });
  const [helpful, reduce] = page.card.querySelectorAll("[data-feedback]");
  reduce.click();
  assert.equal(helpful.attributes.get("aria-pressed"), "true");
  assert.equal(reduce.attributes.get("aria-pressed"), "false");
  assert.equal(storedEvents(storage)["0123456789abcdef01234567"].vote, "helpful");
});

test("pages without a published reference disable feedback", () => {
  const state = runFeedback({ eventRef: "" });
  assert.equal(state.card.querySelectorAll("[data-feedback]").every(b => b.disabled), true);
  assert.match(state.card.querySelector(".feedback-status").textContent, /仅供阅读/);
});
