/**
 * `--format json` prints the Python server's envelope too.
 *
 * Same fixtures as the MCP replay in galaxy-mcp: what the Python server emitted
 * for a set of calls, written down by a generator over there
 * (`uv run python -m tests.envelope_fixtures`) with the Galaxy replies it was
 * answered with. Here each case is driven through the real command line -- flags
 * parsed, op run, `render` printing to stdout -- and the JSON it prints has to
 * parse to the same envelope.
 *
 * Compared: the keys, and `data`, `success`, `count`, `pagination` and `message`
 * exactly. The sentence is part of the contract for every tool -- it is what an
 * agent reads first, and for the nine tools whose page the output budget cuts it
 * also decides where the cut falls, because the budget is measured on the whole
 * envelope and a longer sentence cuts a row. Key order is not part of the contract.
 *
 * Nothing is skipped, the cut cases included. This surface measures the budget
 * on the compact line the other two measure and prints indented afterwards, so
 * a page is cut at the same row wherever it is read.
 *
 * A failure case is compared differently, because this surface's failure shape is its own
 * and not the other server's. There is no FastMCP here to turn a raised exception into an
 * MCP error result, and a command line has to say something an exit code can be read off:
 * so a failure prints `{success: false, message, errorKind}` and what parity means is the
 * SENTENCE -- the same words the other server's tool said, with nothing of FastMCP's
 * wrapper around it. `errorKind` stays, because the exit code hangs on it.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, it, expect, vi } from "vitest";
import { allOperations, createGalaxyContext, type AnyOperation } from "@galaxyproject/galaxy-ops";
import { buildProgram } from "../src/program";
import { classifyField, flagName } from "../src/flags";

const FIXTURES = new URL(
  "../../../../mcp-server-galaxy-py/tests/testdata/envelopes/",
  import.meta.url,
);

interface Route {
  method: string;
  path: string;
  query: Record<string, string>;
  status: number;
  headers: Record<string, string>;
  /** The reply as a value, for a route whose exact bytes do not matter. */
  body?: unknown;
  /** The reply as bytes, for a route that feeds a sentence quoting the body. */
  bodyText?: string;
}

interface CaseEntry {
  tool: string;
  case: string;
  note: string;
  outcome: "success" | "failure";
  input: Record<string, unknown>;
  envelope: string;
  replies: string;
}

/** A failed call on the other server: the wire result, and the tool's own sentence. */
interface FailureCase {
  sentence: string;
}

const readJson = <T>(name: string): T =>
  JSON.parse(readFileSync(fileURLToPath(new URL(name, FIXTURES)), "utf8")) as T;

const index = readJson<{ cases: CaseEntry[] }>("index.json");
const replayable = index.cases.filter((c) => c.outcome === "success");
const failures = index.cases.filter((c) => c.outcome === "failure");

/** The canned replies, matched the way the generator registered them. */
function replier(baseUrl: string, routes: Route[]): typeof fetch {
  return (async (input: unknown, init?: { method?: string }): Promise<Response> => {
    const href =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : (input as { url: string }).url;
    const url = new URL(href);
    // openapi-fetch builds a Request and calls fetch with it, so a write's method is on
    // the Request and not in an init the caller passed -- read `init` alone and every
    // POST, PUT and DELETE in the table is looked up as a GET and answered with a 404.
    const method = (
      init?.method ?? (input instanceof Request ? input.method : "GET")
    ).toUpperCase();
    const target = routes
      .filter((route) => {
        if (route.method.toUpperCase() !== method) return false;
        const want = route.path.startsWith("http")
          ? new URL(route.path)
          : new URL(route.path, baseUrl);
        if (want.origin !== url.origin || want.pathname !== url.pathname) return false;
        return Object.entries(route.query).every(([k, v]) => url.searchParams.get(k) === v);
      })
      .sort((a, b) => Object.keys(b.query).length - Object.keys(a.query).length)[0];
    if (!target) {
      return new Response(JSON.stringify({ err_msg: `no canned reply for ${method} ${href}` }), {
        status: 404,
        headers: { "content-type": "application/json" },
      });
    }
    // 204, 205 and 304 carry no body by definition, and the runtime refuses to build a
    // reply that has both: an empty string is still a body. A route that answers one of
    // them is answered with no body at all, which is also what Galaxy sends.
    const noBody = target.status === 204 || target.status === 205 || target.status === 304;
    return new Response(noBody ? null : (target.bodyText ?? JSON.stringify(target.body)), {
      status: target.status,
      headers: { "content-type": "application/json", ...target.headers },
    });
  }) as typeof fetch;
}

/** The op's own name for a parameter the wire spells in snake_case. */
const wireName = (key: string): string => key.replace(/[A-Z]/g, (m) => `_${m.toLowerCase()}`);

/**
 * The command line that makes this call.
 *
 * Positionals first, so a variadic flag at the end cannot swallow one, and each
 * value spelled the way the CLI takes it -- which is the whole reason this runs
 * through the program rather than calling the op: the flag layer is part of the
 * surface under test.
 */
function argvFor(op: AnyOperation, input: Record<string, unknown>): string[] {
  const positionals: string[] = [];
  const options: string[] = [];
  for (const [key, schema] of Object.entries(op.input)) {
    const wire = wireName(key);
    if (!(wire in input)) continue;
    const value = input[wire];
    // A null is not expressible on a command line, and it does not need to be: the other
    // server drops a parameter that is still None, so leaving the flag off says the same
    // thing the MCP caller's explicit null says. The case then exercises the same call.
    if (value === null) continue;
    const kind = classifyField(schema as never);
    const flag = `--${flagName(key)}`;
    if (kind === "positional") positionals.push(String(value));
    else if (kind === "boolean") options.push(value ? flag : `--no-${flagName(key)}`);
    else if (kind === "array") options.push(flag, ...(value as unknown[]).map(String));
    else if (kind === "json") options.push(flag, JSON.stringify(value));
    else options.push(flag, String(value));
  }
  return [op.name, ...positionals, ...options, "--format", "json"];
}

async function runCli(argv: string[], baseUrl: string, fetchImpl: typeof fetch) {
  const out = vi.spyOn(console, "log").mockImplementation(() => {});
  const err = vi.spyOn(console, "error").mockImplementation(() => {});
  // The context's client takes the canned fetch as an argument; the IWC manifest and
  // quay.io do not -- they go through the global. All of them are answered from the
  // table, so no case here can reach the network.
  vi.stubGlobal("fetch", fetchImpl);
  process.exitCode = 0;
  try {
    await buildProgram({
      // The table's own base, spelled as it spells it: get_server_info answers with the
      // address it was given, so a trailing slash dropped here would be a difference in
      // the harness rather than in the surface.
      makeContext: () => createGalaxyContext({ baseUrl, apiKey: "not-a-key", fetchImpl }),
    }).parseAsync(["node", "galaxy-cli", ...argv]);
    return {
      stdout: out.mock.calls.flat().join(""),
      stderr: err.mock.calls.flat().join(""),
      exitCode: process.exitCode,
    };
  } finally {
    out.mockRestore();
    err.mockRestore();
    vi.unstubAllGlobals();
    process.exitCode = 0;
  }
}

describe("the CLI's json envelope is the Python server's", () => {
  it("has cases to check", () => {
    expect(replayable.length).toBeGreaterThan(40);
  });

  it.each(replayable.map((c) => [`${c.tool} / ${c.case}`, c] as const))(
    "%s",
    async (_label, entry) => {
      const expected = readJson<Record<string, unknown>>(entry.envelope);
      const replies = readJson<{ baseUrl: string; routes: Route[] }>(entry.replies);
      const op = allOperations.find((o) => o.name === entry.tool);
      expect(op, `no op named ${entry.tool}`).toBeDefined();
      // The IWC manifest and the container recommender are both memoised for the life
      // of the process; a case serving its own manifest or its own tag list has to
      // start from nothing remembered.
      const { __resetIwcCacheForTest } = await import("../../galaxy-ops/src/iwc-manifest");
      const { __clearRecommendationCacheForTest } = await import("../../galaxy-ops/src/mulled");
      __resetIwcCacheForTest();
      __clearRecommendationCacheForTest();

      const run = await runCli(
        argvFor(op!, entry.input),
        replies.baseUrl,
        replier(replies.baseUrl, replies.routes),
      );
      expect(run.exitCode, run.stderr || run.stdout).toBe(0);
      const printed = JSON.parse(run.stdout) as Record<string, unknown>;

      expect(Object.keys(printed).sort(), "the envelope's keys").toEqual(
        Object.keys(expected).sort(),
      );
      expect(printed.data, "data").toEqual(expected.data);
      expect(printed.success, "success").toEqual(expected.success);
      expect(printed.count, "count").toEqual(expected.count);
      expect(printed.pagination, "pagination").toEqual(expected.pagination);
      expect(typeof printed.message, "message").toBe("string");
      expect(printed.message, "message").toEqual(expected.message);
    },
  );
});

describe("the CLI's json failure says what the Python server's tool said", () => {
  it("has cases to check", () => {
    expect(failures.length).toBeGreaterThan(10);
  });

  it.each(failures.map((c) => [`${c.tool} / ${c.case}`, c] as const))(
    "%s",
    async (_label, entry) => {
      const expected = readJson<FailureCase>(entry.envelope);
      const replies = readJson<{ baseUrl: string; routes: Route[] }>(entry.replies);
      const op = allOperations.find((o) => o.name === entry.tool);
      expect(op, `no op named ${entry.tool}`).toBeDefined();
      const { __resetIwcCacheForTest } = await import("../../galaxy-ops/src/iwc-manifest");
      const { __clearRecommendationCacheForTest } = await import("../../galaxy-ops/src/mulled");
      __resetIwcCacheForTest();
      __clearRecommendationCacheForTest();

      const run = await runCli(
        argvFor(op!, entry.input),
        replies.baseUrl,
        replier(replies.baseUrl, replies.routes),
      );
      const printed = JSON.parse(run.stdout) as Record<string, unknown>;
      expect(printed.success, "success").toBe(false);
      expect(printed.message, "the sentence").toEqual(expected.sentence);
      // The shape is this surface's own, and it is the whole of it: no data, no count, no
      // pagination, and a kind for the exit code to be read off.
      expect(Object.keys(printed).sort(), "the failure's keys").toEqual([
        "errorKind",
        "message",
        "success",
      ]);
      expect(run.exitCode, "the exit code").not.toBe(0);
    },
  );
});

/**
 * The table view reads its hint from the new place. It went through
 * pagination.helperText until the wire keys became the Python server's, and a
 * renderer left behind would simply print nothing rather than fail.
 */
describe("the table view still says how to page on", () => {
  it("prints the helper text on stderr, from helper_text", async () => {
    const entry = index.cases.find(
      (c) => c.tool === "list_history_ids" && c.case === "full_page",
    )!;
    const replies = readJson<{ baseUrl: string; routes: Route[] }>(entry.replies);
    const expected = readJson<{ pagination: { helper_text: string } }>(entry.envelope);
    const op = allOperations.find((o) => o.name === entry.tool)!;
    const argv = argvFor(op, entry.input).slice(0, -2); // drop --format json: table is the default

    const run = await runCli(argv, replies.baseUrl, replier(replies.baseUrl, replies.routes));
    expect(run.stderr).toContain(expected.pagination.helper_text);
    // And the rows are a table of the page itself, not the shape around it.
    expect(run.stdout).toMatch(/id\s+name/);
  });

  it("tables a history's contents, which arrive under their own key", async () => {
    const entry = index.cases.find(
      (c) => c.tool === "get_history_contents" && c.case === "full_page",
    )!;
    const replies = readJson<{ baseUrl: string; routes: Route[] }>(entry.replies);
    const op = allOperations.find((o) => o.name === entry.tool)!;
    const argv = argvFor(op, entry.input).slice(0, -2);

    const run = await runCli(argv, replies.baseUrl, replier(replies.baseUrl, replies.routes));
    expect(run.stdout).toMatch(/id\s+hid/);
    expect(run.stdout).not.toContain("contents");
  });
});
