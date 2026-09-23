import { describe, it, expect } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { allOperations } from "@galaxyproject/galaxy-ops";
import { buildServer, toolNames, toolAnnotations, annotationsFor } from "../src/server";

describe("MCP surface is a mechanical projection", () => {
  it("registers one tool per registered op", () => {
    const names = toolNames();
    expect(names).toContain("get_user");
    expect(names).toContain("get_histories");
    expect(names).toContain("create_history");
    expect(names).toContain("get_tool_details");
  });

  it("marks read ops readOnly and writes not", () => {
    const ann = toolAnnotations();
    expect(ann.get_histories?.readOnlyHint).toBe(true);
    expect(ann.create_history?.readOnlyHint).toBe(false);
    expect(ann.create_history?.destructiveHint).toBe(false);
    expect(ann.run_tool?.readOnlyHint).toBe(false); // executes a tool
  });

  it("derives per-op annotations from readOnly/destructive hints", () => {
    expect(annotationsFor({})).toEqual({ readOnlyHint: true, destructiveHint: false });
    expect(annotationsFor({ readOnly: false })).toEqual({ readOnlyHint: false, destructiveHint: false });
    expect(annotationsFor({ readOnly: false, destructive: true })).toEqual({
      readOnlyHint: false,
      destructiveHint: true,
    });
  });

  it("tells a model what an op needs from the server, in the op's own description", async () => {
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const server = buildServer({ baseUrl: "https://g.example", apiKey: "K" });
    const client = new Client({ name: "surface-check", version: "0" });
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
    try {
      const { tools } = await client.listTools();
      const byName = new Map(tools.map((t) => [t.name, t.description ?? ""]));
      for (const op of allOperations) {
        const described = byName.get(op.name) ?? "";
        expect(described.startsWith(op.summary), op.name).toBe(true);
        expect(described.includes("Requires Galaxy"), op.name).toBe(op.requires !== undefined);
      }
      expect(byName.get("list_page_revisions")).toContain("Requires Galaxy 26.1 or newer.");
      // get_page works on 26.0, so it must not pick the sentence up.
      expect(byName.get("get_page")).not.toContain("Requires Galaxy");
    } finally {
      await client.close();
      await server.close();
    }
  });

  it("builds a server without throwing", () => {
    expect(buildServer({ baseUrl: "https://g.example", apiKey: "K" })).toBeDefined();
  });
});
