/**
 * The page behind each MCP Apps card: a compact list the user can page, collapse and open.
 *
 * The host renders the page in a sandboxed frame when a `show_*` tool returns, and hands it that
 * tool's arguments and result. The card never talks to Galaxy and holds no credential: "Show
 * more" first reveals rows it already has, then asks the host to call the card's read tool for
 * the next page (from the result's `next_offset`, with the arguments the card was opened with).
 * The page has no external scripts, styles or images.
 *
 * Protocol: https://github.com/modelcontextprotocol/ext-apps (`ui/initialize`,
 * `ui/notifications/tool-input` and `tool-result`, `tools/call`, `ui/request-display-mode`,
 * `ui/resource-teardown`, `ui/notifications/size-changed`).
 */
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";

const MIME_TYPE = "text/html;profile=mcp-app";

export interface CardSpec {
  uri: string;
  title: string;
  /** The read tool "Show more" calls once the rows the card holds run out. Absent: nothing to fetch. */
  pageTool?: string;
  /** The tool Retry calls, with the opening call's arguments, when that call failed. */
  openTool?: string;
  /** Arguments of the opening call to repeat on every page, by their wire names. */
  keep: string[];
  /**
   * Page-side JavaScript: `(envelope, kept) => ({ items, total, nextOffset, scope, caption })`.
   * `total` and `nextOffset` are null when unknown or when there is nothing more to fetch.
   */
  read: string;
  /**
   * Page-side JavaScript: `(item) => ({ name, meta: [string], state, detail: [[label, value]] })`.
   * Rows without `detail` cannot be opened.
   */
  row: string;
  empty: string;
}

const STEP = 5;
const FETCH = 20;

export function cardPage(card: CardSpec): string {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${card.title}</title>
<style>
  :root {
    color-scheme: light dark;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
    --text: light-dark(#1d2233, #e7eaf3);
    --secondary: light-dark(#5d6478, #a0a8bd);
    --surface: light-dark(#fff, #1b2030);
    --row: light-dark(#eef0f5, #232a3d);
    --line: light-dark(#d5d9e3, #333b52);
    --accent: light-dark(#2f4bb8, #8ea4ff);
    --ok: light-dark(#1d7a46, #6fd39a);
    --bad: light-dark(#b42c2c, #ff8a8a);
    --busy: light-dark(#8a5a00, #f0c060);
    --note: light-dark(#eee9fb, #2c2547);
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 4px; color: var(--text); background: transparent; font-size: 14px; }
  p { margin: 0; }
  .card { border: 1px solid var(--line); border-radius: 14px; background: var(--surface); overflow: hidden; }
  header { display: flex; align-items: center; gap: 10px; padding: 10px 12px; }
  .mark { flex: none; width: 22px; height: 22px; border-radius: 6px; background: #f2c200; color: #1d2233; display: grid; place-items: center; font-weight: 700; font-size: 12px; }
  .heading { display: grid; min-width: 0; flex: 1; }
  .title { overflow: hidden; font-weight: 650; text-overflow: ellipsis; white-space: nowrap; }
  .scope { color: var(--secondary); font-size: 12px; }
  .body { display: grid; gap: 8px; padding: 8px 12px 12px; border-top: 1px solid var(--line); }
  .note { padding: 7px 10px; border-radius: 8px; background: var(--note); }
  .note b { margin-right: 6px; }
  ul { display: grid; gap: 4px; margin: 0; padding: 0; list-style: none; }
  .row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; align-items: center; width: 100%; padding: 6px 8px; border: 0; border-radius: 8px; background: var(--row); color: inherit; font: inherit; text-align: left; }
  button.row { cursor: pointer; }
  button.row:hover { outline: 1px solid var(--accent); }
  .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .meta { display: block; color: var(--secondary); font-size: 12px; }
  .state { font-size: 11px; font-weight: 600; }
  .state.ok { color: var(--ok); }
  .state.error, .state.failed { color: var(--bad); }
  .state.running, .state.queued, .state.new { color: var(--busy); }
  .error { color: var(--bad); }
  .actions { display: flex; flex-wrap: wrap; gap: 8px; }
  button.link { padding: 3px 11px; border: 1px solid var(--line); border-radius: 999px; background: transparent; color: var(--accent); font: inherit; font-size: 12px; cursor: pointer; }
  button.link:disabled { color: var(--secondary); cursor: default; }
  button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  dl { display: grid; grid-template-columns: 7em minmax(0, 1fr); gap: 3px 10px; margin: 0; }
  dt { color: var(--secondary); }
  dd { margin: 0; overflow-wrap: anywhere; }
</style>
</head>
<body>
<main class="card" id="card"></main>
<script>
  const STEP = ${STEP};
  const FETCH = ${FETCH};
  const PAGE_TOOL = ${JSON.stringify(card.pageTool ?? null)};
  const OPEN_TOOL = ${JSON.stringify(card.openTool ?? null)};
  const KEEP = ${JSON.stringify(card.keep)};
  const TITLE = ${JSON.stringify(card.title)};
  const EMPTY = ${JSON.stringify(card.empty)};
  const read = ${card.read};
  const rowOf = ${card.row};
  const root = document.getElementById("card");
  const pending = new Map();
  let nextId = 1;
  let canCallTools = false;
  let displayModes = [];
  let displayMode = "inline";
  // What the card holds and shows. Showing fewer rows never forgets rows or moves the cursor.
  const kept = {};
  let opening = {}; // the opening call's arguments, for Retry
  let view = null; // { items, total, nextOffset, scope, caption } once a result arrives
  let shown = STEP;
  let collapsed = false;
  let opened = null; // index of the row whose detail is open
  let loading = false;
  let failure = null;

  function post(message) {
    window.parent.postMessage(message, "*");
  }

  function request(method, params) {
    const id = nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        pending.delete(id);
        reject(new Error("The host did not answer."));
      }, 30000);
      pending.set(id, { resolve, reject, timer });
      post({ jsonrpc: "2.0", id, method, params });
    });
  }

  window.addEventListener("message", (event) => {
    if (event.source !== window.parent) return;
    const message = event.data;
    if (message?.jsonrpc !== "2.0") return;
    if (message.method === undefined && pending.has(message.id)) {
      const { resolve, reject, timer } = pending.get(message.id);
      pending.delete(message.id);
      clearTimeout(timer);
      if (message.error) reject(new Error(message.error.message || "The host refused the request."));
      else resolve(message.result);
    } else if (message.method === "ui/notifications/tool-input") {
      const args = message.params?.arguments ?? {};
      opening = args;
      for (const key of KEEP) if (args[key] !== undefined) kept[key] = args[key];
    } else if (message.method === "ui/notifications/tool-result") {
      show(message.params);
    } else if (message.method === "ui/resource-teardown") {
      post({ jsonrpc: "2.0", id: message.id, result: {} });
    } else if (message.method === "ui/notifications/host-context-changed") {
      applyHost(message.params);
      render();
    }
  });

  /** A tool result's envelope: the JSON text this server answers with. */
  function envelope(result) {
    if (result?.isError) throw new Error(result.content?.[0]?.text || "Galaxy returned an error.");
    const text = result?.content?.find((block) => block.type === "text")?.text;
    try {
      return JSON.parse(text);
    } catch {
      // A host can hand over its own note in place of a result it found too large to pass on.
      throw new Error(text ? text.slice(0, 300) : "The result did not reach the card.");
    }
  }

  function el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else if (key === "aria") for (const [name, v] of Object.entries(value)) node.setAttribute("aria-" + name, v);
      else node[key] = value;
    }
    node.append(...children.filter((child) => child !== null && child !== undefined && child !== false));
    return node;
  }

  /** Galaxy writes UTC times without a zone. */
  function galaxyTime(value) {
    if (!value) return "";
    const time = new Date(/[zZ]|[+-]\\d\\d:?\\d\\d$/.test(value) ? value : value + "Z");
    return Number.isNaN(time.getTime()) ? "" : time.toLocaleString();
  }

  function bytes(value) {
    if (typeof value !== "number") return "";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let n = value;
    let unit = 0;
    while (n >= 1024 && unit < units.length - 1) { n /= 1024; unit++; }
    return (unit === 0 ? n : n.toFixed(1)) + " " + units[unit];
  }

  function counts() {
    if (!view) return "";
    const held = view.items.length;
    const total = typeof view.total === "number" ? view.total : view.nextOffset === null ? held : held + "+";
    return "Showing " + Math.min(shown, held) + " of " + total;
  }

  function show(result) {
    try {
      view = read(envelope(result), kept);
      failure = null;
      shown = Math.min(STEP, view.items.length);
    } catch (error) {
      view = null;
      failure = error.message;
    }
    opened = null;
    render();
  }

  async function reopen() {
    loading = true;
    render();
    try {
      show(await request("tools/call", { name: OPEN_TOOL, arguments: opening }));
    } catch (error) {
      failure = error.message;
    } finally {
      loading = false;
      render();
    }
  }

  async function showMore() {
    if (shown < view.items.length) {
      shown = Math.min(view.items.length, shown + STEP);
      return render();
    }
    if (!PAGE_TOOL || view.nextOffset === null || !canCallTools) return;
    loading = true;
    failure = null;
    render();
    try {
      const page = read(envelope(await request("tools/call", {
        name: PAGE_TOOL,
        arguments: { ...kept, limit: FETCH, offset: view.nextOffset },
      })), kept);
      view.items.push(...page.items);
      view.nextOffset = page.nextOffset;
      if (typeof page.total === "number") view.total = page.total;
      shown = Math.min(view.items.length, shown + STEP);
    } catch (error) {
      failure = error.message;
    } finally {
      loading = false;
      render();
    }
  }

  async function toggleFullscreen() {
    try {
      const result = await request("ui/request-display-mode", { mode: displayMode === "fullscreen" ? "inline" : "fullscreen" });
      if (result?.mode) displayMode = result.mode;
    } catch {
      // The host keeps its current mode; nothing to undo.
    }
    render();
  }

  function render() {
    const header = el("header", {},
      el("span", { className: "mark", aria: { hidden: "true" } }, "G"),
      el("div", { className: "heading" },
        el("span", { className: "title" }, TITLE),
        el("span", { className: "scope" }, view ? [view.scope, counts()].filter(Boolean).join(" \\u00b7 ") : failure ? "Could not load" : "Waiting for Galaxy\\u2026"),
      ),
      displayModes.includes("fullscreen")
        ? el("button", { type: "button", className: "link", onclick: toggleFullscreen }, displayMode === "fullscreen" ? "Exit full screen" : "Full screen")
        : null,
      el("button", { type: "button", className: "link", aria: { expanded: String(!collapsed) }, onclick: () => { collapsed = !collapsed; render(); } }, collapsed ? "Expand" : "Collapse"),
    );
    if (collapsed) return root.replaceChildren(header);
    root.replaceChildren(header, el("div", { className: "body" }, ...(opened === null ? list() : detail())));
  }

  function list() {
    if (!view) {
      if (!failure) return [];
      const retry = OPEN_TOOL && canCallTools;
      return [
        el("p", { className: "error" }, failure),
        retry ? el("div", { className: "actions" }, el("button", { type: "button", className: "link", disabled: loading, onclick: reopen }, loading ? "Loading\\u2026" : "Retry")) : null,
      ];
    }
    const rows = view.items.slice(0, shown).map((item, index) => {
      const row = rowOf(item);
      const content = [
        el("span", { className: "name" }, row.name, el("span", { className: "meta" }, row.meta.filter(Boolean).join(" \\u00b7 "))),
        el("span", { className: "state " + (row.state || "") }, row.state || ""),
      ];
      return el("li", {}, row.detail
        ? el("button", { type: "button", className: "row", onclick: () => { opened = index; render(); } }, ...content)
        : el("div", { className: "row" }, ...content));
    });
    const more = shown < view.items.length || (PAGE_TOOL && view.nextOffset !== null && canCallTools);
    const actions = [
      shown > STEP ? el("button", { type: "button", className: "link", onclick: () => { shown = Math.max(STEP, shown - STEP); render(); } }, "Show less") : null,
      more || failure ? el("button", { type: "button", className: "link", disabled: loading, onclick: showMore }, loading ? "Loading\\u2026" : failure ? "Retry" : "Show more") : null,
    ].filter(Boolean);
    return [
      view.caption ? el("p", { className: "note" }, el("b", {}, "Assistant note"), view.caption) : null,
      rows.length ? el("ul", {}, ...rows) : el("p", { className: "meta" }, EMPTY),
      failure ? el("p", { className: "error" }, failure) : null,
      actions.length ? el("div", { className: "actions" }, ...actions) : null,
    ];
  }

  function detail() {
    const row = rowOf(view.items[opened]);
    return [
      el("div", { className: "actions" }, el("button", { type: "button", className: "link", onclick: () => { opened = null; render(); } }, "Back")),
      el("b", {}, row.name),
      el("dl", {}, ...row.detail.filter(([, value]) => value !== "" && value !== undefined && value !== null).flatMap(([label, value]) => [el("dt", {}, label), el("dd", {}, String(value))])),
    ];
  }

  function applyHost(context) {
    if (!context) return;
    if (Array.isArray(context.availableDisplayModes)) displayModes = context.availableDisplayModes;
    if (context.displayMode) displayMode = context.displayMode;
    if (context.theme === "light" || context.theme === "dark") document.documentElement.style.colorScheme = context.theme;
  }

  render();
  request("ui/initialize", {
    appInfo: { name: TITLE, version: "1" },
    appCapabilities: { availableDisplayModes: ["inline", "fullscreen"] },
    protocolVersion: "2026-01-26",
  }).then((result) => {
    canCallTools = Boolean(result?.hostCapabilities?.serverTools);
    applyHost(result?.hostContext);
    render();
    post({ jsonrpc: "2.0", method: "ui/notifications/initialized", params: {} });
    // Sizes are reported once the host has the card, starting now; collapsing shrinks it.
    new ResizeObserver(() => {
      post({ jsonrpc: "2.0", method: "ui/notifications/size-changed", params: { height: Math.ceil(document.documentElement.getBoundingClientRect().height) } });
    }).observe(document.documentElement);
  }, (error) => {
    failure = error.message;
    render();
  });
</script>
</body>
</html>
`;
}

/** Serve a card's page. A tool shows it by naming its URI in `_meta.ui.resourceUri`. */
export function registerCard(server: McpServer, name: string, card: CardSpec): void {
  const html = cardPage(card);
  const meta = { ui: { prefersBorder: false } };
  server.registerResource(name, card.uri, { title: card.title, mimeType: MIME_TYPE, _meta: meta }, (uri) => ({
    contents: [{ uri: uri.href, mimeType: MIME_TYPE, text: html, _meta: meta }],
  }));
}
