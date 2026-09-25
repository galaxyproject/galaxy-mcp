import { describe, it, expect } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { allOperations, requirementSentence } from "@galaxyproject/galaxy-ops";
import { buildServer, toolNames, toolAnnotations, annotationsFor, toolResult } from "../src/server";

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
        // The bound it names, not merely that it names one. The parity check reads this
        // side's requirement off the op, because MCP has no field for it, and that is
        // only honest while the sentence a client is shown says the same thing.
        if (op.requires) {
          expect(described, op.name).toContain(requirementSentence(op.requires.galaxy));
        }
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

/**
 * What a model is handed for one call, pinned here because the ops derive every
 * page cap from a reconstruction of it (`galaxy-ops/test/util/byte-budget.ts`)
 * and cannot import this function. If a second block, structured content or any
 * other wrapping is ever added, those caps are measuring a payload that no longer
 * exists and this test is where that gets caught.
 */
describe("the text a tool call puts on the wire", () => {
  const envelope = { data: [{ id: "h1" }], success: true, message: "1", pagination: { total: 1 } };

  it("is the whole envelope, compact, in a single text block", () => {
    const result = toolResult(envelope);
    expect(result.content).toHaveLength(1);
    expect(result.content[0]).toEqual({ type: "text", text: JSON.stringify(envelope) });
    expect(Object.keys(result).sort()).toEqual(["content", "isError"]);
  });

  it("flags a failed envelope as an error without changing the text", () => {
    const failed = { data: undefined, success: false, message: "nope", errorKind: "validation" };
    expect(toolResult(failed)).toEqual({
      content: [{ type: "text", text: JSON.stringify(failed) }],
      isError: true,
    });
  });
});
