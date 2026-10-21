import { describe, expect, it } from "vitest";
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
