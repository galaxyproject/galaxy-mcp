/**
 * `pnpm cards:preview`: serves the cards on this machine, framed by the stand-in host and fed the
 * sample results, at http://127.0.0.1:5179 (or PORT). Each page is built when it is asked for, and
 * the script restarts when a source file changes, so a reload shows the cards as the code has them.
 */
import { createServer } from "node:http";
import { cardPage } from "../../src/apps/card-page";
import { datasetsCard, historiesCard, historyCard } from "../../src/apps/cards";
import { toolResult } from "../../src/server";
import { hostPage } from "./host";
import { answer, openings } from "./sample";

const cards = {
  histories: { card: historiesCard, label: "show_histories" },
  history: { card: historyCard, label: "show_history" },
  datasets: { card: datasetsCard, label: "show_datasets" },
};

const port = Number(process.env.PORT ?? 5179);

createServer(async (request, response) => {
  const send = (status: number, type: string, body: string) => {
    response.writeHead(status, { "content-type": type, "cache-control": "no-store" });
    response.end(body);
  };
  const path = new URL(request.url ?? "/", "http://preview").pathname;
  const shown = path.startsWith("/card/") ? cards[path.slice("/card/".length) as keyof typeof cards] : undefined;
  if (request.method === "GET" && path === "/") {
    send(200, "text/html; charset=utf-8", hostPage(Object.entries(cards).map(([name, { label }]) => ({ name, label, ...openings[name]! }))));
  } else if (request.method === "GET" && shown) {
    send(200, "text/html; charset=utf-8", cardPage(shown.card, "usegalaxy.example"));
  } else if (request.method === "POST" && path === "/call") {
    let body = "";
    for await (const chunk of request) body += chunk;
    const { name, arguments: args } = JSON.parse(body) as { name: string; arguments?: Record<string, unknown> };
    send(200, "application/json", JSON.stringify(toolResult(name, answer(name, args ?? {}))));
  } else {
    send(404, "text/plain", "Not found");
  }
}).listen(port, "127.0.0.1", () => console.log(`Cards preview: http://127.0.0.1:${port}`));
