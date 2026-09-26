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

/**
 * One CLI run, judged on its own. Each run gets its own console capture and its own exit status,
 * because a shared spy lets an earlier success answer for a later failure and the next test's
 * reset hides the exit code.
 */
async function runCli(argv: string[], makeContext: () => ReturnType<typeof createGalaxyContext>) {
  const out = vi.spyOn(console, "log").mockImplementation(() => {});
  const err = vi.spyOn(console, "error").mockImplementation(() => {});
  process.exitCode = 0;
  try {
    await buildProgram({ makeContext }).parseAsync(["node", "galaxy-cli", ...argv]);
    return {
      stdout: out.mock.calls.flat().join(""),
      stderr: err.mock.calls.flat().join(""),
      exitCode: process.exitCode,
    };
  } finally {
    out.mockRestore();
    err.mockRestore();
  }
}

/** Answers the invocation detail route with one record and the index with a list, recording each URL. */
function invocationsContext(asked: string[]) {
  return () =>
    createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      fetchImpl: (async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
        asked.push(url);
        const body = url.includes("/api/invocations/")
          ? { id: "inv1", state: "scheduled" }
          : [{ id: "inv1", state: "scheduled" }];
        return new Response(JSON.stringify(body), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }) as typeof fetch,
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

  it("takes get_invocations' id as a flag now that it is optional", async () => {
    // It used to be a required positional, so `galaxy-cli get_invocations abc123` worked. The
    // parity change made it optional, which moves it to --invocation-id.
    const asked: string[] = [];
    const run = await runCli(["get_invocations", "--invocation-id", "inv1", "--format", "json"], invocationsContext(asked));
    expect(asked).toEqual([expect.stringContaining("/api/invocations/inv1")]);
    expect(run.stdout).toContain('"success": true');
    expect(run.exitCode).toBe(0);
  });

  it("lists invocations when get_invocations is given no id", async () => {
    const asked: string[] = [];
    const run = await runCli(["get_invocations", "--format", "json"], invocationsContext(asked));
    expect(asked).toEqual([expect.stringContaining("/api/invocations?")]);
    expect(asked[0]).not.toContain("workflow_id=");
    expect(run.stdout).toContain('"success": true');
    expect(run.exitCode).toBe(0);
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
