/**
 * An MCP Apps card listing the user's histories.
 *
 * The host renders this page in a sandboxed frame when `show_histories` returns, and hands it
 * that tool's result. "Load more" calls `get_histories` back through the host for the next page,
 * so the card never talks to Galaxy itself and holds no credential. The page has no external
 * scripts, styles or images; everything it needs is in this string.
 *
 * Protocol: https://github.com/modelcontextprotocol/ext-apps (`ui/initialize`,
 * `ui/notifications/tool-result`, `tools/call`, `ui/notifications/size-changed`).
 */
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";

export const HISTORIES_APP_URI = "ui://galaxy/histories.html";
const MIME_TYPE = "text/html;profile=mcp-app";
const PAGE_SIZE = 5;

export const historiesAppHtml = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Galaxy histories</title>
<style>
  :root {
    color-scheme: light dark;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
    --text: light-dark(#1d1d1f, #f5f5f7);
    --secondary: light-dark(#6e6e73, #a1a1a6);
    --surface: light-dark(#fff, #1c1c1e);
    --row: light-dark(rgba(0, 0, 0, .03), rgba(255, 255, 255, .05));
    --line: light-dark(rgba(0, 0, 0, .08), rgba(255, 255, 255, .11));
    --accent: light-dark(#0071e3, #2997ff);
    --danger: light-dark(#d70015, #ff453a);
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 8px; color: var(--text); background: transparent; }
  .card { padding: 16px; border: 1px solid var(--line); border-radius: 16px; background: var(--surface); }
  h1 { margin: 0 0 4px; font-size: 1.05rem; font-weight: 650; }
  #status { margin: 0 0 12px; color: var(--secondary); font-size: .85rem; }
  #status[data-error] { color: var(--danger); }
  ul { display: grid; gap: 6px; margin: 0; padding: 0; list-style: none; }
  li { min-width: 0; padding: 10px 12px; border-radius: 10px; background: var(--row); }
  .name { overflow: hidden; font-size: .92rem; font-weight: 600; text-overflow: ellipsis; white-space: nowrap; }
  .meta { margin-top: 2px; color: var(--secondary); font-size: .78rem; }
  button { margin-top: 12px; padding: 7px 14px; border: 1px solid var(--line); border-radius: 999px; color: var(--accent); background: transparent; font: inherit; font-size: .82rem; cursor: pointer; }
  button:disabled { color: var(--secondary); cursor: default; }
</style>
</head>
<body>
<main class="card">
  <h1>Galaxy histories</h1>
  <p id="status" role="status">Waiting for Galaxy…</p>
  <ul id="histories"></ul>
  <button id="more" type="button" hidden>Load more</button>
</main>
<script>
  const PAGE_SIZE = ${PAGE_SIZE};
  const statusLine = document.getElementById("status");
  const list = document.getElementById("histories");
  const more = document.getElementById("more");
  const pending = new Map();
  let nextId = 1;
  let canCallTools = false;
  let nextOffset = null;
  let total = null;

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
    if (message.id !== undefined && pending.has(message.id)) {
      const { resolve, reject, timer } = pending.get(message.id);
      pending.delete(message.id);
      clearTimeout(timer);
      if (message.error) reject(new Error(message.error.message || "The host refused the request."));
      else resolve(message.result);
    } else if (message.method === "ui/notifications/tool-result") {
      show(message.params, false);
    } else if (message.method === "ui/notifications/host-context-changed") {
      applyTheme(message.params?.theme);
    }
  });

  /** A tool result's envelope: the JSON text this server answers with. */
  function envelope(result) {
    if (result?.isError) throw new Error(result.content?.[0]?.text || "Galaxy returned an error.");
    const text = result?.content?.find((block) => block.type === "text")?.text;
    return JSON.parse(text);
  }

  function show(result, append) {
    let page;
    try {
      page = envelope(result);
    } catch (error) {
      setStatus(error.message, true);
      return;
    }
    if (!append) list.replaceChildren();
    for (const history of page.data || []) list.append(row(history));
    total = page.pagination?.total_items ?? (page.data || []).length;
    nextOffset = page.pagination?.has_next ? page.pagination.next_offset : null;
    setStatus(list.children.length === 0 ? "No histories yet." : "Showing " + list.children.length + " of " + total + ".");
    more.hidden = nextOffset === null || !canCallTools;
  }

  function row(history) {
    const item = document.createElement("li");
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = history.name || "Unnamed history";
    const meta = document.createElement("div");
    meta.className = "meta";
    const parts = [];
    if (typeof history.count === "number") parts.push(history.count + (history.count === 1 ? " item" : " items"));
    // Galaxy writes UTC times without a zone.
    const updated = history.update_time ? new Date(/[zZ]|[+-]\\d\\d:?\\d\\d$/.test(history.update_time) ? history.update_time : history.update_time + "Z") : null;
    if (updated && !Number.isNaN(updated.getTime())) parts.push("updated " + updated.toLocaleString());
    meta.textContent = parts.join(" \\u00b7 ");
    item.append(name, meta);
    return item;
  }

  function setStatus(text, error = false) {
    statusLine.textContent = text;
    statusLine.toggleAttribute("data-error", error);
  }

  more.addEventListener("click", async () => {
    more.disabled = true;
    more.textContent = "Loading\\u2026";
    try {
      const result = await request("tools/call", {
        name: "get_histories",
        arguments: { limit: PAGE_SIZE, offset: nextOffset },
      });
      show(result, true);
    } catch (error) {
      setStatus(error.message, true);
    } finally {
      more.disabled = false;
      more.textContent = "Load more";
    }
  });

  function applyTheme(theme) {
    if (theme === "light" || theme === "dark") document.documentElement.style.colorScheme = theme;
  }

  new ResizeObserver(() => {
    post({ jsonrpc: "2.0", method: "ui/notifications/size-changed", params: { height: Math.ceil(document.documentElement.getBoundingClientRect().height) } });
  }).observe(document.documentElement);

  request("ui/initialize", {
    appInfo: { name: "Galaxy histories", version: "1" },
    appCapabilities: {},
    protocolVersion: "2026-01-26",
  }).then((result) => {
    canCallTools = Boolean(result?.hostCapabilities?.serverTools);
    applyTheme(result?.hostContext?.theme);
    more.hidden = nextOffset === null || !canCallTools;
    post({ jsonrpc: "2.0", method: "ui/notifications/initialized", params: {} });
  }, (error) => setStatus(error.message, true));
</script>
</body>
</html>
`;

/** Serve the card. A tool shows it by naming HISTORIES_APP_URI in `_meta.ui.resourceUri`. */
export function registerHistoriesApp(server: McpServer): void {
  const meta = { ui: { prefersBorder: false } };
  server.registerResource(
    "galaxy-histories",
    HISTORIES_APP_URI,
    { title: "Galaxy histories", mimeType: MIME_TYPE, _meta: meta },
    (uri) => ({ contents: [{ uri: uri.href, mimeType: MIME_TYPE, text: historiesAppHtml, _meta: meta }] }),
  );
}
