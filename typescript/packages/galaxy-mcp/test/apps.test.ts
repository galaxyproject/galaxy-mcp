import { describe, expect, it } from "vitest";
import { runInNewContext } from "node:vm";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { createGalaxyContext } from "@galaxyproject/galaxy-ops";
import { HISTORIES_APP_URI, historiesAppHtml } from "../src/apps/histories";
import { buildServer, toolNames } from "../src/server";

const histories = [
  { id: "h1", name: "RNA-seq", count: 4, update_time: "2026-10-21T12:00:00.000000" },
  { id: "h2", name: "ChIP-seq", count: 9, update_time: "2026-10-20T12:00:00.000000" },
];
// A Galaxy that only knows its history list.
const ctx = createGalaxyContext({
  baseUrl: "http://galaxy.invalid",
  apiKey: "k",
  fetchImpl: async (request) => {
    const url = new URL(request instanceof Request ? request.url : String(request));
    if (url.pathname === "/api/histories") return Response.json(histories);
    return new Response("not found", { status: 404 });
  },
});

async function connect(apps: boolean): Promise<Client> {
  const [clientSide, serverSide] = InMemoryTransport.createLinkedPair();
  await buildServer(ctx, { apps }).connect(serverSide);
  const client = new Client({ name: "apps-test", version: "0" });
  await client.connect(clientSide);
  return client;
}

describe("MCP Apps cards", () => {
  it("are off by default, so the surface is the operations and nothing else", async () => {
    const client = await connect(false);
    const { tools } = await client.listTools();
    expect(tools.map((t) => t.name).sort()).toEqual(toolNames().sort());
    expect(client.getServerCapabilities()?.resources).toBeUndefined();
  });

  it("add show_histories, pointing at the card it opens", async () => {
    const client = await connect(true);
    const { tools } = await client.listTools();
    expect(tools.map((t) => t.name).sort()).toEqual([...toolNames(), "show_histories"].sort());
    const show = tools.find((t) => t.name === "show_histories")!;
    const read = tools.find((t) => t.name === "get_histories")!;
    expect(show._meta).toEqual({ ui: { resourceUri: HISTORIES_APP_URI } });
    expect(show.inputSchema).toEqual(read.inputSchema);
    expect(show.annotations).toEqual(read.annotations);
    expect(read._meta).toBeUndefined();
  });

  it("serve the card as an MCP App", async () => {
    const client = await connect(true);
    const { contents } = await client.readResource({ uri: HISTORIES_APP_URI });
    expect(contents).toEqual([expect.objectContaining({ uri: HISTORIES_APP_URI, mimeType: "text/html;profile=mcp-app", text: historiesAppHtml })]);
  });

  it("answer show_histories exactly as get_histories", async () => {
    const client = await connect(true);
    const args = { limit: 1, offset: 0 };
    const shown = await client.callTool({ name: "show_histories", arguments: args });
    const read = await client.callTool({ name: "get_histories", arguments: args });
    expect(shown).toEqual(read);
    expect(JSON.parse((shown.content as { text: string }[])[0]!.text).data).toEqual([histories[0]]);
  });
});

describe("the histories card page", () => {
  const script = historiesAppHtml.match(/<script>([\s\S]*)<\/script>/)?.[1] ?? "";

  it("is one self-contained page", () => {
    expect(historiesAppHtml).toMatch(/^<!doctype html>/);
    // Nothing loaded from elsewhere: the host's sandbox would block it, and the card needs none.
    expect(historiesAppHtml).not.toMatch(/\b(src|href)=|@import|url\(/);
  });

  it("carries a script that parses", () => {
    expect(script).toContain('request("ui/initialize"');
    expect(() => new Function(script)).not.toThrow();
  });

  it("reads Galaxy's zone-less times as UTC", () => {
    const pattern = /[zZ]|[+-]\d\d:?\d\d$/;
    expect(script).toContain(String.raw`/[zZ]|[+-]\d\d:?\d\d$/`);
    expect(pattern.test("2026-10-21T12:00:00.123456")).toBe(false);
    expect(pattern.test("2026-10-21T12:00:00Z")).toBe(true);
    expect(pattern.test("2026-10-21T12:00:00+02:00")).toBe(true);
  });
});

/** Runs the card's script against a stand-in for the few DOM pieces it touches, as a host would. */
function runCard() {
  class Element {
    textContent = "";
    hidden = false;
    disabled = false;
    className = "";
    children: Element[] = [];
    handlers: Record<string, () => unknown> = {};
    append(...children: Element[]) {
      this.children.push(...children);
    }
    replaceChildren() {
      this.children = [];
    }
    toggleAttribute() {}
    addEventListener(type: string, handler: () => unknown) {
      this.handlers[type] = handler;
    }
  }
  const elements: Record<string, Element> = { status: new Element(), histories: new Element(), more: new Element() };
  elements.more!.hidden = true;
  const sent: Record<string, any>[] = [];
  const parent = { postMessage: (message: Record<string, any>) => sent.push(message) };
  const listeners: ((event: { source: unknown; data: unknown }) => void)[] = [];
  const documentElement = { style: {}, getBoundingClientRect: () => ({ height: 120 }) };
  const script = historiesAppHtml.match(/<script>([\s\S]*)<\/script>/)![1]!;
  runInNewContext(script, {
    window: { parent, addEventListener: (type: string, f: (typeof listeners)[number]) => type === "message" && listeners.push(f) },
    document: { getElementById: (id: string) => elements[id], createElement: () => new Element(), documentElement },
    ResizeObserver: class {
      constructor(private readonly report: () => void) {}
      observe() {
        this.report();
      }
    },
    setTimeout,
    clearTimeout,
  });
  const settle = () => new Promise((resolve) => setTimeout(resolve, 0));
  return {
    sent,
    elements,
    deliver: async (data: Record<string, unknown>) => {
      for (const listener of listeners) listener({ source: parent, data: { jsonrpc: "2.0", ...data } });
      await settle();
    },
    /** Answer the card's ui/initialize as a host with server tools. */
    initialize: async function () {
      await this.deliver({ id: sent[0]!.id, result: { hostCapabilities: { serverTools: {} } } });
    },
  };
}

const page = (data: unknown[], next: number | null) => ({
  content: [{ type: "text", text: JSON.stringify({ success: true, data, pagination: { total_items: 3, has_next: next !== null, next_offset: next } }) }],
});

describe("the histories card's script", () => {
  it("starts with ui/initialize and reports its size only once the host has it", async () => {
    const card = runCard();
    expect(card.sent.map((m) => m.method)).toEqual(["ui/initialize"]);
    await card.initialize();
    expect(card.sent.map((m) => m.method)).toEqual(["ui/initialize", "ui/notifications/initialized", "ui/notifications/size-changed"]);
  });

  it("answers the host's teardown request, even under an id its own request uses", async () => {
    const card = runCard();
    const ownId = card.sent[0]!.id;
    await card.deliver({ id: ownId, method: "ui/resource-teardown", params: {} });
    expect(card.sent.at(-1)).toEqual({ jsonrpc: "2.0", id: ownId, result: {} });
    await card.initialize(); // its own request is still answerable
    expect(card.sent.at(-2)?.method).toBe("ui/notifications/initialized");
  });

  it("keeps the name filter it was opened with when it loads more", async () => {
    const card = runCard();
    await card.initialize();
    await card.deliver({ method: "ui/notifications/tool-input", params: { arguments: { name: "RNA-seq", limit: 1 } } });
    await card.deliver({ method: "ui/notifications/tool-result", params: page([histories[0]], 1) });
    expect(card.elements.histories!.children).toHaveLength(1);
    expect(card.elements.more!.hidden).toBe(false);

    void card.elements.more!.handlers.click!();
    const call = card.sent.at(-1)!;
    expect(call).toMatchObject({ method: "tools/call", params: { name: "get_histories", arguments: { name: "RNA-seq", limit: 5, offset: 1 } } });
    await card.deliver({ id: call.id, result: page([histories[1]], null) });
    expect(card.elements.histories!.children).toHaveLength(2);
    expect(card.elements.more!.hidden).toBe(true);
  });
});
