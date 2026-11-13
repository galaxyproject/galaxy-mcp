import { describe, expect, it } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { createGalaxyContext, type GalaxyContext } from "@galaxyproject/galaxy-ops";
import { buildServer } from "../src/server";

/**
 * A scoped context reaches the MCP surface unchanged: the host builds it with the run's scope
 * and hands it to buildServer, and a refusal arrives the way every other refusal does -- an
 * MCP error carrying the sentence -- while an account read is cut down before it is serialised.
 */
const user = { id: "u1", username: "alice", email: "alice@example.org", preferences: { theme: "dark" } };

function scopedContext(): GalaxyContext {
  return createGalaxyContext({
    baseUrl: "http://galaxy.invalid",
    apiKey: "k",
    scope: { historyIds: ["h1"], blockedOperations: ["create_history"] },
    fetchImpl: (async (req: Request) => {
      if (new URL(req.url).pathname.endsWith("/api/users/current")) {
        return new Response(JSON.stringify(user), { status: 200, headers: { "content-type": "application/json" } });
      }
      throw new Error(`no Galaxy in this test: ${req.url}`);
    }) as typeof fetch,
  });
}

async function connect(ctx: GalaxyContext): Promise<Client> {
  const [clientSide, serverSide] = InMemoryTransport.createLinkedPair();
  await buildServer(ctx).connect(serverSide);
  const client = new Client({ name: "scope-test", version: "0" });
  await client.connect(clientSide);
  return client;
}

describe("a server over a scoped context", () => {
  it("refuses a history outside the scope as an MCP error, before any request", async () => {
    const client = await connect(scopedContext());
    const result = await client.callTool({ name: "get_history_details", arguments: { history_id: "h2" } });
    expect(result.isError).toBe(true);
    expect(result.content).toEqual([
      {
        type: "text",
        text:
          "Error calling tool 'get_history_details': history_id 'h2' is outside this run's scope. " +
          "Histories this run may read: h1. Nothing was sent to Galaxy.",
      },
    ]);
  });

  it("refuses a blocked tool by name", async () => {
    const client = await connect(scopedContext());
    const result = await client.callTool({ name: "create_history", arguments: { history_name: "new" } });
    expect(result.isError).toBe(true);
    expect(JSON.stringify(result.content)).toContain("create_history is not available in this run");
  });

  it("answers get_user with the id and username only", async () => {
    const client = await connect(scopedContext());
    const result = await client.callTool({ name: "get_user", arguments: {} });
    expect(result.isError).toBe(false);
    const text = (result.content as Array<{ text: string }>)[0]!.text;
    expect(JSON.parse(text).data).toEqual({ id: "u1", username: "alice" });
    expect(text).not.toContain("email");
    expect(text).not.toContain("preferences");
  });
});
