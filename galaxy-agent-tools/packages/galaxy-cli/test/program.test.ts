import { describe, it, expect, vi, beforeEach } from "vitest";
import { buildProgram } from "../src/program";
import { createGalaxyContext } from "@galaxyproject/galaxy-ops";

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
    expect(names.length).toBe(14);
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
