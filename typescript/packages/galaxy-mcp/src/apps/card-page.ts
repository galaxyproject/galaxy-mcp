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
   * Page-side JavaScript:
   * `(item) => ({ name, number, kind, meta: [string], state, side: [string], detail: [[label, value]] })`.
   * `number` is the item's place in its history, shown before the name. `kind` is a few letters for
   * the tile that leads the row; rows without one have no tile. `meta` goes under the name; `state`,
   * Galaxy's own word for it, and `side` go at the row's end. Rows without `detail` cannot be
   * opened.
   */
  row: string;
  empty: string;
}

const STEP = 5;
const FETCH = 20;

/**
 * A Galaxy's address as a card shows it: the host, and any path the Galaxy is served under.
 * Nothing for an address that is not a URL, which a card then leaves out.
 */
export function galaxyLabel(baseUrl: string | undefined): string | undefined {
  try {
    const url = new URL(baseUrl ?? "");
    return url.host + url.pathname.replace(/\/+$/, "");
  } catch {
    return undefined;
  }
}

/** The page for one card. `galaxy` names the Galaxy it shows, under its title. */
export function cardPage(card: CardSpec, galaxy?: string): string {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${card.title}</title>
<style>
  /* Quiet and neutral: colour only for a state. */
  :root {
    color-scheme: light dark;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", "Helvetica Neue", sans-serif;
    --text: light-dark(#1d1d1f, #f5f5f7);
    --secondary: light-dark(#6e6e73, #a1a1a6);
    --surface: light-dark(rgba(255, 255, 255, .94), rgba(28, 28, 30, .96));
    --row: light-dark(rgba(0, 0, 0, .025), rgba(255, 255, 255, .045));
    --hover: light-dark(rgba(0, 0, 0, .04), rgba(255, 255, 255, .06));
    --line: light-dark(rgba(0, 0, 0, .08), rgba(255, 255, 255, .11));
    --strong: light-dark(rgba(0, 0, 0, .16), rgba(255, 255, 255, .2));
    --good: light-dark(#248a3d, #30d158);
    --busy: light-dark(#9a6700, #ffd60a);
    --bad: light-dark(#d70015, #ff453a);
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 6px; color: var(--text); background: transparent; }
  p { margin: 0; }
  .card { padding: 14px; border: 1px solid var(--line); border-radius: 12px; background: var(--surface); box-shadow: 0 1px 2px light-dark(rgba(0, 0, 0, .04), rgba(0, 0, 0, .3)); overflow: hidden; }
  header { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .brand { display: flex; align-items: center; gap: 10px; min-width: 0; }
  .mark { flex: none; display: grid; place-items: center; width: 34px; height: 34px; border: 1px solid var(--line); border-radius: 10px; background: #fff; }
  .mark svg { display: block; width: 26px; height: auto; }
  .heading { display: grid; min-width: 0; }
  .title { font-size: 1.02rem; font-weight: 600; letter-spacing: -.025em; line-height: 1.25; overflow-wrap: anywhere; }
  .origin { margin-top: 1px; color: var(--secondary); font-size: .76rem; overflow-wrap: anywhere; }
  .header-actions { display: flex; flex: none; align-items: center; gap: 8px; }
  .status, .message { margin-top: 10px; color: var(--secondary); font-size: .82rem; line-height: 1.4; }
  .note { margin-top: 10px; padding: 8px 10px; border-radius: 8px; background: var(--row); font-size: .82rem; line-height: 1.4; }
  .note b { margin-right: 8px; font-weight: 600; }
  ul { margin: 10px 0 0; padding: 0; border-top: 1px solid var(--line); list-style: none; }
  li { min-width: 0; border-bottom: 1px solid var(--line); }
  .row { display: flex; align-items: center; gap: 12px; width: 100%; min-width: 0; padding: 7px 4px; border: 0; border-radius: 8px; background: transparent; color: inherit; font: inherit; text-align: left; }
  button.row { cursor: pointer; }
  button.row:hover { background: var(--hover); }
  .kind { flex: none; display: grid; place-items: center; width: 32px; height: 32px; border-radius: 8px; background: var(--row); color: var(--secondary); font: 600 .6rem/1 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .02em; }
  .name { flex: 1; min-width: 0; overflow: hidden; font-size: .92rem; font-weight: 550; letter-spacing: -.012em; line-height: 1.3; text-overflow: ellipsis; white-space: nowrap; }
  .hid { margin-right: 6px; color: var(--secondary); font-weight: 400; font-variant-numeric: tabular-nums; }
  .meta { display: block; margin-top: 1px; overflow: hidden; color: var(--secondary); font-size: .75rem; font-weight: 400; letter-spacing: 0; text-overflow: ellipsis; }
  .side { flex: none; display: grid; gap: 1px; color: var(--secondary); font-size: .72rem; line-height: 1.25; text-align: right; overflow-wrap: anywhere; }
  .state { font-size: .76rem; font-weight: 500; }
  /* A state in the colour Galaxy's history panel gives it: done, at work, failed. Waiting stays plain. */
  .state.ok { color: var(--good); }
  .state.running, .state.setting_metadata, .state.upload { color: var(--busy); }
  .state.error, .state.failed, .state.discarded { color: var(--bad); }
  p.error { margin-top: 10px; color: var(--bad); font-size: .82rem; line-height: 1.4; }
  .actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
  .actions.paging { justify-content: center; }
  button.link { min-height: 28px; padding: 3px 10px; border: 1px solid var(--strong); border-radius: 8px; background: var(--surface); color: var(--text); font: inherit; font-size: .78rem; font-weight: 500; cursor: pointer; }
  button.link:hover:not(:disabled) { background: var(--hover); }
  button.link:disabled { color: var(--secondary); background: var(--row); cursor: default; }
  button:focus-visible { outline: 2px solid var(--text); outline-offset: 2px; }
  .opened { display: block; margin-top: 10px; font-size: .92rem; font-weight: 550; }
  dl { display: grid; grid-template-columns: 7em minmax(0, 1fr); gap: 4px 12px; margin: 10px 0 0; font-size: .86rem; }
  dt { color: var(--secondary); }
  dd { margin: 0; overflow-wrap: anywhere; font-variant-numeric: tabular-nums; }
  /* In a narrow frame the header's buttons drop under the title, and a row's end keeps to its share. */
  @media (max-width: 520px) {
    body { padding: 4px; }
    .card { padding: 12px; border-radius: 14px; }
    header { flex-wrap: wrap; gap: 8px; }
    .row { align-items: flex-start; }
    .side { max-width: 42%; }
  }
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
  const GALAXY = ${JSON.stringify(galaxy ?? null)};
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
  function parseTime(value) {
    if (!value) return null;
    const time = new Date(/[zZ]|[+-]\\d\\d:?\\d\\d$/.test(value) ? value : value + "Z");
    return Number.isNaN(time.getTime()) ? null : time;
  }

  function galaxyTime(value) {
    return parseTime(value)?.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) ?? "";
  }

  /** The day alone, for a row. */
  function galaxyDate(value) {
    return parseTime(value)?.toLocaleDateString(undefined, { dateStyle: "medium" }) ?? "";
  }

  function bytes(value) {
    if (typeof value !== "number") return "";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let n = value;
    let unit = 0;
    while (n >= 1024 && unit < units.length - 1) { n /= 1024; unit++; }
    return (unit === 0 ? n : n.toFixed(1)) + " " + units[unit];
  }

  /** The tile for a dataset: its datatype in a few letters, without what follows a sequence format. */
  function kindOf(extension) {
    const format = String(extension || "").toLowerCase();
    if (/^fast[aq]/.test(format)) return format.slice(0, 5).toUpperCase();
    return format === "tabular" ? "TABLE" : format.slice(0, 6).toUpperCase() || "DATA";
  }

  /**
   * Galaxy's mark, drawn here because the page carries no image: two bars and a third that fades
   * out, as in the project's own logo (galaxyLogo.duotone.svg in galaxyproject/galaxy).
   */
  function mark() {
    const shape = (tag, attributes) => {
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
      return node;
    };
    const bar = (x, y, width, fill, opacity = 1) => shape("rect", { x, y, width, height: 145, rx: 12, fill, opacity });
    const svg = shape("svg", { viewBox: "0 0 591 469", "aria-hidden": "true" });
    svg.append(
      bar(0, 0, 512, "#58585a"), bar(0, 162, 418, "#58585a"), bar(187, 324, 190, "#d1be04"),
      ...[0.75, 0.55, 0.38, 0.24, 0.12].map((opacity, step) => bar(365 + step * 44, 324, 50, "#d1be04", opacity)),
    );
    return el("span", { className: "mark" }, svg);
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
      el("div", { className: "brand" },
        mark(),
        el("div", { className: "heading" },
          el("span", { className: "title" }, TITLE),
          GALAXY ? el("span", { className: "origin" }, GALAXY) : null,
        ),
      ),
      el("div", { className: "header-actions" },
        displayModes.includes("fullscreen")
          ? el("button", { type: "button", className: "link", onclick: toggleFullscreen }, displayMode === "fullscreen" ? "Exit full screen" : "Full screen")
          : null,
        el("button", { type: "button", className: "link", aria: { expanded: String(!collapsed) }, onclick: () => { collapsed = !collapsed; render(); } }, collapsed ? "Expand" : "Collapse"),
      ),
    );
    // What is shown, and how much of it: kept when the card is collapsed.
    const status = el("p", { className: "status" },
      view ? [view.scope, counts()].filter(Boolean).join(" \\u00b7 ") : failure ? "Could not load" : "Waiting for Galaxy\\u2026");
    if (collapsed) return root.replaceChildren(header, status);
    root.replaceChildren(header, status, ...(opened === null ? list() : detail()).filter(Boolean));
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
      const meta = row.meta.filter(Boolean).join(" \\u00b7 ");
      const content = [
        row.kind ? el("span", { className: "kind", aria: { hidden: "true" } }, row.kind) : null,
        el("span", { className: "name" }, numbered(row), row.name, meta ? el("span", { className: "meta" }, meta) : null),
        el("span", { className: "side" },
          row.state ? el("span", { className: "state " + row.state }, row.state) : null,
          ...(row.side || []).filter(Boolean).map((line) => el("span", {}, line)),
        ),
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
      rows.length ? el("ul", {}, ...rows) : el("p", { className: "message" }, EMPTY),
      failure ? el("p", { className: "error" }, failure) : null,
      actions.length ? el("div", { className: "actions paging" }, ...actions) : null,
    ];
  }

  /** An item's number in its history, before its name, as Galaxy lists it. */
  function numbered(row) {
    return typeof row.number === "number" ? el("span", { className: "hid" }, row.number + ":") : null;
  }

  function detail() {
    const row = rowOf(view.items[opened]);
    return [
      el("div", { className: "actions" }, el("button", { type: "button", className: "link", onclick: () => { opened = null; render(); } }, "Back")),
      el("b", { className: "opened" }, numbered(row), row.name),
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
export function registerCard(server: McpServer, name: string, card: CardSpec, galaxy?: string): void {
  const html = cardPage(card, galaxy);
  const meta = { ui: { prefersBorder: false } };
  server.registerResource(name, card.uri, { title: card.title, mimeType: MIME_TYPE, _meta: meta }, (uri) => ({
    contents: [{ uri: uri.href, mimeType: MIME_TYPE, text: html, _meta: meta }],
  }));
}
