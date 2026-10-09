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
  "../../../../contract/envelopes/",
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

/**
 * The exit code each failure case ends with.
 *
 * A CLI-only fact, which is why it lives here and not in the fixture the Python server
 * writes: that server has no command line, so there is nothing over there to be in parity
 * with. What the exit code IS, though, is the only part of a failure a script reads without
 * parsing anything, so "non-zero" is not an assertion -- it passes equally for a usage error
 * reported as a service outage. Every case names its number, and the guard below refuses to
 * let a case exist without one.
 *
 * Read off `errorKind` through `exitCodeFor`, so a row here is a claim about the kind too.
 * Three rows are a claim I would rather not be making: `create_user_tool`'s argument refusals
 * are worded and raised before any request goes out, and still report `connection` -- the
 * thing `errors.ts` argues against -- where `recommend_biocontainer`'s equally local refusal
 * reports `validation`. That inconsistency is older than the failure envelope and not one of
 * the things this round set out to change, so it is written down rather than quietly fixed;
 * the table is what will make the change visible when it happens.
 */
const EXIT_CODES: Record<string, number> = {
  // EX_USAGE 64 -- validation: the caller has to change something, and a retry cannot help
  "get_dataset_details/is_a_collection": 64,
  "get_history_contents/limit_below_one": 64,
  "get_job_details/both_ids_given": 64,
  "get_job_details/neither_id_given": 64,
  "get_invocations/limit_above_galaxys_cap": 64,
  "get_workflow_input_template/negative_version": 64,
  "list_history_ids/history_without_an_id": 64,
  "list_jobs/limit_below_one": 64,
  "list_workflows/limit_above_the_ceiling": 64,
  "recommend_biocontainer/package_entry_with_no_name": 64,
  "run_tool/refused_over_credentials": 64,
  "run_tool/refused_over_the_inputs": 64,
  "run_tool/refused_with_nothing_readable": 64,
  "run_user_tool/refused_over_the_inputs": 64,
  "run_user_tool/refused_with_a_reply_about_credentials": 64,
  "search_tools_by_name/negative_offset": 64,
  "update_history/every_field_null": 64,
  "update_history/nothing_to_update": 64,
  // EX_NOINPUT 66 -- not_found: a 404, or an id nothing answers to
  "cancel_workflow_invocation/not_found": 66,
  "delete_user_tool/not_found": 66,
  "delete_user_tool/not_found_with_an_empty_body": 66,
  "get_collection_details/not_found": 66,
  "get_history_details/not_found": 66,
  "get_invocations/malformed_id": 66,
  "get_invocations/unknown_id": 66,
  "get_iwc_workflow_details/no_such_trs_id": 66,
  "get_job_details/dataset_not_found": 66,
  "get_job_details/job_id_malformed": 66,
  "get_job_details/job_not_found": 66,
  "get_job_details/no_job_made_this_dataset": 66,
  "get_page/not_found": 66,
  "get_page_revision/not_found": 66,
  "get_tool_citations/not_found": 66,
  "get_tool_details/unknown_tool_at_a_version": 66,
  "get_tool_details/version_not_installed": 66,
  "get_tool_input_template/not_found": 66,
  "get_tool_panel/no_such_section": 66,
  "get_workflow_details/not_found": 66,
  "import_workflow_from_iwc/no_such_trs_id": 66,
  "list_page_revisions/not_found": 66,
  "revert_page_revision/not_found": 66,
  "update_page/not_found": 66,
  // EX_UNAVAILABLE 69 -- connection: the server answered something we cannot act on
  "create_history/refused_by_galaxy": 69,
  "create_page/refused_by_galaxy": 69,
  "create_page/report_without_a_title": 69,
  "create_user_tool/container_is_not_a_string": 69,
  "create_user_tool/refused_by_galaxy": 69,
  "create_user_tool/representation_missing_a_field": 69,
  "create_user_tool/wrong_class": 69,
  "get_histories/server_error": 69,
  "get_invocations/error_body_under_a_200": 69,
  "get_iwc_workflow_details/manifest_refused": 69,
  "get_iwc_workflows/manifest_refused": 69,
  "get_job_details/the_job_read_refused": 69,
  "get_server_info/configuration_refused": 69,
  "get_tool_run_examples/refused_without_a_hint": 69,
  "get_workflow_details/server_error": 69,
  "get_workflow_input_template/server_error": 69,
  "get_workflow_input_template/version_out_of_range": 69,
  "import_workflow_from_iwc/manifest_refused": 69,
  "import_workflow_from_iwc/refused_by_galaxy": 69,
  "invoke_workflow/refused_by_galaxy": 69,
  "list_history_ids/server_error": 69,
  "list_jobs/malformed_history_id": 69,
  "list_user_tools/server_error": 69,
  "recommend_iwc_workflows/manifest_refused": 69,
  "search_iwc_workflows/manifest_refused": 69,
  "search_tools_by_keywords/server_error": 69,
  "update_history/refused_by_galaxy": 69,
  // EX_PROTOCOL 76 -- version: the server is too old and nothing was sent
  "create_page/galaxy_too_old": 76,
  // EX_NOPERM 77 -- auth: a 401 or a 403
  "delete_user_tool/permission_denied_with_an_empty_body": 77,
  "get_collection_details/permission_denied": 77,
  "get_history_contents/permission_denied": 77,
  "get_tool_details/unauthorized": 77,
  "get_user/unauthorized": 77,
  "list_jobs/history_not_accessible": 77,
  "list_pages/permission_denied": 77,
  "run_tool/permission_denied": 77,
  "run_user_tool/lookup_refused": 77,
};

describe("the CLI's json failure says what the Python server's tool said", () => {
  it("has cases to check", () => {
    expect(failures.length).toBeGreaterThan(10);
  });

  it("knows the exit code for every failure case, and names no case that is gone", () => {
    const cases = failures.map((c) => `${c.tool}/${c.case}`).sort();
    expect(Object.keys(EXIT_CODES).sort()).toEqual(cases);
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
      expect(run.exitCode, "the exit code").toBe(EXIT_CODES[`${entry.tool}/${entry.case}`]);
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
