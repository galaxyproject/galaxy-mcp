import { describe, it, expect, vi } from "vitest";
import { buildProgram } from "../src/program";
import { createGalaxyContext } from "@galaxyproject/galaxy-ops";

/**
 * The same decoding the MCP surface applies to an incoming call, reached the way a person
 * reaches it. Commander hands every argument over as a string, so the string half of
 * pydantic's integer table is what a command line can actually produce -- `--limit 5`,
 * `--limit 5.0`, `--limit 1_000` -- and the rest of the table is asserted on the other
 * surface, where JSON can carry a number or a boolean.
 *
 * Booleans are declarative flags here (`--published`, `--no-published`), so commander gives
 * a real boolean and the word list has nothing to do; it matters for MCP, not for this.
 */
const ACCEPTED: Array<[string, number]> = [
  ["5", 5],
  [" 5 ", 5],
  ["+5", 5],
  ["-5", -5],
  ["05", 5],
  ["5.0", 5],
  ["1_000", 1000],
];

const REFUSED = ["5.5", "1e2", "0x10", "", "abc", "٥"];

/** Answers any GET with an empty list, recording the URLs, and 26.1 for the version guard. */
function recordingContext(asked: string[]) {
  return () =>
    createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      fetchImpl: (async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
        asked.push(url);
        const body = url.includes("/api/version") ? { version_major: "26.1", version_minor: "26.1.1" } : [];
        return new Response(JSON.stringify(body), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }) as typeof fetch,
    });
}

async function runCli(argv: string[], asked: string[]) {
  const out = vi.spyOn(console, "log").mockImplementation(() => {});
  const err = vi.spyOn(console, "error").mockImplementation(() => {});
  process.exitCode = 0;
  try {
    await buildProgram({ makeContext: recordingContext(asked) }).parseAsync([
      "node",
      "galaxy-cli",
      ...argv,
    ]);
    return {
      stderr: err.mock.calls.flat().join(""),
      exitCode: process.exitCode,
    };
  } finally {
    out.mockRestore();
    err.mockRestore();
  }
}

const query = (asked: string[], route: string): string => {
  const hit = asked.find((url) => url.includes(route));
  expect(hit, `nothing asked for ${route}`).toBeDefined();
  return hit!;
};

describe("a numeric flag, decoded the way the other surface decodes it", () => {
  it.each(ACCEPTED)("takes --offset %j as %i", async (given, want) => {
    const asked: string[] = [];
    const run = await runCli(["list_pages", "--offset", given, "--format", "json"], asked);
    expect(run.exitCode, run.stderr).toBe(0);
    expect(query(asked, "/api/pages")).toContain(`offset=${want}`);
  });

  it.each(ACCEPTED.filter(([, want]) => want >= 1 && want <= 500))(
    "takes --limit %j as %i on a paged listing",
    async (given, want) => {
      const asked: string[] = [];
      const run = await runCli(["list_pages", "--limit", given, "--format", "json"], asked);
      expect(run.exitCode, run.stderr).toBe(0);
      expect(query(asked, "/api/pages")).toContain(`limit=${want}`);
    },
  );

  it.each(REFUSED)("exits 64 for --offset %j, without calling Galaxy", async (given) => {
    const asked: string[] = [];
    const run = await runCli(["list_pages", "--offset", given, "--format", "json"], asked);
    expect(run.exitCode).toBe(64);
    expect(asked.some((url) => url.includes("/api/pages"))).toBe(false);
  });

  it("takes a converted version through to the query string", async () => {
    const asked: string[] = [];
    const run = await runCli(["get_workflow_details", "w1", "--version", "3", "--format", "json"], asked);
    expect(run.exitCode, run.stderr).toBe(0);
    expect(query(asked, "/api/workflows/w1")).toContain("version=3");
  });

  it("still refuses a window the op itself will not take, as a usage error", async () => {
    // Decoding and validating are different jobs: "0" is a number here and not a page there,
    // so it gets through the flag and is refused by the op with the sentence Python uses.
    const asked: string[] = [];
    const out = vi.spyOn(console, "log").mockImplementation(() => {});
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    process.exitCode = 0;
    try {
      await buildProgram({ makeContext: recordingContext(asked) }).parseAsync([
        "node",
        "galaxy-cli",
        "list_history_ids",
        "--limit",
        "0",
      ]);
      expect(process.exitCode).toBe(64);
      expect(out.mock.calls.flat().join("") + err.mock.calls.flat().join("")).toContain(
        "limit must be at least 1 (got 0)",
      );
    } finally {
      out.mockRestore();
      err.mockRestore();
    }
  });
});
