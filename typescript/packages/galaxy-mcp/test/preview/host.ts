/**
 * A stand-in for an MCP Apps host: one page that frames each card and does the host's part of
 * the protocol for it, so the cards can be looked at in a browser with no host and no Galaxy.
 *
 * It answers `ui/initialize`, hands a card the call that opened it and that call's result, relays
 * the card's own `tools/call` to `/call`, grants full screen, and sizes each frame to what its card
 * reports. Each card is handed to its frame as a document, with scripts allowed and nothing
 * else, which is how a host shows the resource it read from the server.
 */

export interface PreviewedCard {
  /** The card's path under `/card/`. */
  name: string;
  label: string;
  /** The call that opens it. */
  tool: string;
  args: Record<string, unknown>;
}

export function hostPage(cards: PreviewedCard[]): string {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Galaxy cards</title>
<style>
  :root { color-scheme: light dark; font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif; }
  body { margin: 0; padding: 16px; background: light-dark(#ececf0, #111113); color: light-dark(#1d1d1f, #f5f5f7); }
  header { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 20px; margin-bottom: 16px; }
  h1 { margin: 0; font-size: 16px; }
  label { display: inline-flex; align-items: center; gap: 6px; }
  select { font: inherit; }
  main { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 16px; }
  section { width: var(--width, 480px); max-width: 100%; }
  section.fullscreen { position: fixed; inset: 0; z-index: 1; width: auto; padding: 16px; overflow: auto; background: light-dark(#ececf0, #111113); }
  h2 { margin: 0 0 6px; font-size: 13px; font-weight: 600; }
  iframe { display: block; width: 100%; height: 120px; border: 0; }
</style>
</head>
<body>
<header>
  <h1>Galaxy cards, on sample data</h1>
  <label>Theme <select id="theme"><option value="light">Light</option><option value="dark">Dark</option></select></label>
  <label>Width <select id="width"><option value="320">320 px</option><option value="480" selected>480 px</option><option value="720">720 px</option></select></label>
</header>
<main id="cards"></main>
<script>
  const CARDS = ${JSON.stringify(cards)};
  const theme = document.getElementById("theme");
  const width = document.getElementById("width");
  const framed = [];
  theme.value = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";

  async function call(name, args) {
    const response = await fetch("/call", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name, arguments: args }) });
    return response.json();
  }

  for (const card of CARDS) {
    const section = document.createElement("section");
    const title = document.createElement("h2");
    const frame = document.createElement("iframe");
    title.textContent = card.label;
    frame.title = card.label;
    frame.setAttribute("sandbox", "allow-scripts");
    section.append(title, frame);
    document.getElementById("cards").append(section);

    const send = (message) => frame.contentWindow.postMessage({ jsonrpc: "2.0", ...message }, "*");
    const context = () => ({
      theme: theme.value,
      displayMode: section.classList.contains("fullscreen") ? "fullscreen" : "inline",
      availableDisplayModes: ["inline", "fullscreen"],
    });
    framed.push({ send, context });

    addEventListener("message", async (event) => {
      if (event.source !== frame.contentWindow) return;
      const message = event.data || {};
      if (message.method === "ui/initialize") {
        send({ id: message.id, result: { protocolVersion: "2026-01-26", hostInfo: { name: "card preview", version: "0" }, hostCapabilities: { serverTools: {} }, hostContext: context() } });
      } else if (message.method === "ui/notifications/initialized") {
        send({ method: "ui/notifications/tool-input", params: { arguments: card.args } });
        send({ method: "ui/notifications/tool-result", params: await call(card.tool, card.args) });
      } else if (message.method === "tools/call") {
        send({ id: message.id, result: await call(message.params.name, message.params.arguments) });
      } else if (message.method === "ui/request-display-mode") {
        section.classList.toggle("fullscreen", message.params.mode === "fullscreen");
        send({ id: message.id, result: { mode: message.params.mode } });
      } else if (message.method === "ui/notifications/size-changed") {
        frame.style.height = message.params.height + "px";
      }
    });
    fetch("/card/" + card.name).then((page) => page.text()).then((page) => { frame.srcdoc = page; });
  }

  function apply() {
    document.documentElement.style.colorScheme = theme.value;
    document.documentElement.style.setProperty("--width", width.value + "px");
    for (const { send, context } of framed) send({ method: "ui/notifications/host-context-changed", params: context() });
  }
  theme.addEventListener("change", apply);
  width.addEventListener("change", apply);
  apply();
</script>
</body>
</html>
`;
}
