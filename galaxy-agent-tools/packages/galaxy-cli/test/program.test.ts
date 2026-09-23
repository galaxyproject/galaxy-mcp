import { describe, it, expect, vi, beforeEach } from "vitest";
import { buildProgram } from "../src/program";
import { allOperations, createGalaxyContext } from "@galaxyproject/galaxy-ops";

function ctxFactory() {
  // a context whose client returns canned data for any GET
  return createGalaxyContext({
    baseUrl: "https://g.example",
    apiKey: "K",
    fetchImpl: (async () =>
      new Response(JSON.stringify([{ id: "h1", name: "alpha" }]), { status: 200, headers: { "content-type": "application/json" } })) as typeof fetch,
  });
}

describe("buildProgram", () => {
  beforeEach(() => { process.exitCode = 0; });

  it("registers one subcommand per op", () => {
    const program = buildProgram({ makeContext: ctxFactory });
    const names = program.commands.map((c) => c.name());
    expect(names).toContain("get_histories");
    expect(names).toContain("create_history");
    expect(names.length).toBe(allOperations.length);
  });

  it("says in --help what an op needs from the server", () => {
    const program = buildProgram({ makeContext: ctxFactory });
    const described = new Map(program.commands.map((c) => [c.name(), c.description()]));
    for (const op of allOperations) {
      const text = described.get(op.name) ?? "";
      expect(text.startsWith(op.summary), op.name).toBe(true);
      expect(text.includes("Requires Galaxy"), op.name).toBe(op.requires !== undefined);
    }
    expect(described.get("list_page_revisions")).toContain("Requires Galaxy 26.1 or newer.");
    // get_page works on 26.0, so it must not pick the sentence up.
    expect(described.get("get_page")).not.toContain("Requires Galaxy");
  });

  it("runs an op and renders json to stdout", async () => {
    const out = vi.spyOn(console, "log").mockImplementation(() => {});
    const program = buildProgram({ makeContext: ctxFactory });
    await program.parseAsync(["node", "galaxy-cli", "get_histories", "--format", "json"]);
    expect(out.mock.calls.flat().join("")).toContain('"success": true');
    expect(process.exitCode === 0 || process.exitCode === undefined).toBe(true);
    out.mockRestore();
  });
});
