/**
 * The MCP surface answers with the Python server's envelope, case for case.
 *
 * The fixtures under mcp-server-galaxy-py/tests/testdata/envelopes are what that
 * server emitted for a set of calls, written down by a generator over there
 * (`uv run python -m tests.envelope_fixtures`) alongside the Galaxy replies it was
 * answered with. Each case is replayed here through the real MCP call path -- a
 * client, a transport, tools/call, the text content block -- against those same
 * replies, and what comes back has to be the same envelope.
 *
 * What "the same" means, exactly:
 *
 * * the same keys, so a surface cannot quietly drop `count` or send `pagination`
 *   where the other sends null;
 * * `data`, `success`, `count` and `pagination` deep-equal, nulls included;
 * * `message` deep-equal for the nine tools whose page the output budget cuts:
 *   get_iwc_workflows, get_tool_panel, list_history_ids, list_user_tools,
 *   list_workflows, recommend_iwc_workflows, search_iwc_workflows,
 *   search_tools_by_keywords and search_tools_by_name. The budget is measured on
 *   the whole envelope, message included, so a sentence eleven bytes longer on
 *   one side cuts one row fewer there -- the prose is part of the contract for
 *   exactly these nine and a fixture that stops at a boundary proves it.
 * * `message` a non-empty string, but NOT compared, for every other tool:
 *   get_histories, get_history_contents, get_invocations, get_tool_citations,
 *   get_tool_run_examples, list_page_revisions and list_pages. Nothing about
 *   those answers depends on the sentence's length, and aligning the rest of the
 *   prose is still its own piece of work.
 *
 * Key ORDER is not part of the contract. JSON objects are unordered, no client
 * depends on it, and neither side promises one.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, it, expect, afterEach, vi } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { buildServer } from "../src/server";
// The IWC manifest is memoised for the life of the process, as it is on the other
// side, so a case that serves its own manifest has to start from nothing
// remembered. Reached directly because it is a test hook and not part of the
// package's public surface.
import { __resetIwcCacheForTest } from "../../galaxy-ops/src/iwc-manifest";

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
  body: unknown;
}

interface CaseEntry {
  tool: string;
  case: string;
  note: string;
  input: Record<string, unknown>;
  envelope: string;
  replies: string;
}

const readJson = <T>(name: string): T =>
  JSON.parse(readFileSync(fileURLToPath(new URL(name, FIXTURES)), "utf8")) as T;

const index = readJson<{ caseCount: number; cases: CaseEntry[] }>("index.json");

/** The tools whose message is part of the contract, because their budget is. */
const MESSAGE_PARITY = new Set([
  "get_iwc_workflows",
  "get_tool_panel",
  "list_history_ids",
  "list_user_tools",
  "list_workflows",
  "recommend_iwc_workflows",
  "search_iwc_workflows",
  "search_tools_by_keywords",
  "search_tools_by_name",
]);

/**
 * Answer a request from the case's table.
 *
 * The same rule the generator registers its mocks under: method and path must
 * match, every query parameter a route declares must match, and the most specific
 * matching route wins. The two implementations do not make the same requests --
 * bioblend asks Galaxy for a window and then counts unpaged, this side fetches
 * once and slices -- so the table answers questions rather than replaying a log.
 */
function replier(baseUrl: string, routes: Route[]): typeof fetch {
  type FetchInput = Parameters<typeof fetch>[0];
  type FetchInit = Parameters<typeof fetch>[1];
  return (async (input: FetchInput, init?: FetchInit): Promise<Response> => {
    const href =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : (input as Request).url;
    const url = new URL(href);
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
    return new Response(JSON.stringify(target.body), {
      status: target.status,
      headers: { "content-type": "application/json", ...target.headers },
    });
  }) as typeof fetch;
}

/** Call one tool the way a client does, and hand back the parsed text block. */
async function callThroughMcp(
  baseUrl: string,
  tool: string,
  args: Record<string, unknown>,
): Promise<{ parsed: Record<string, unknown>; isError: boolean }> {
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  const server = buildServer({ baseUrl, apiKey: "envelope-parity-not-a-key" });
  const client = new Client({ name: "envelope-parity", version: "0" });
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
  try {
    const result = (await client.callTool({ name: tool, arguments: args })) as {
      content: Array<{ type: string; text?: string }>;
      isError?: boolean;
    };
    const text = result.content.map((c) => c.text ?? "").join("");
    return { parsed: JSON.parse(text) as Record<string, unknown>, isError: result.isError === true };
  } finally {
    await client.close();
    await server.close();
  }
}

afterEach(() => {
  vi.unstubAllGlobals();
  __resetIwcCacheForTest();
});

describe("the MCP envelope is the Python server's", () => {
  it("has cases to check", () => {
    expect(index.cases.length).toBe(index.caseCount);
    expect(index.cases.length).toBeGreaterThan(40);
  });

  it.each(index.cases.map((c) => [`${c.tool} / ${c.case}`, c] as const))(
    "%s",
    async (_label, entry) => {
      const expected = readJson<Record<string, unknown>>(entry.envelope);
      const replies = readJson<{ baseUrl: string; routes: Route[] }>(entry.replies);
      __resetIwcCacheForTest();
      vi.stubGlobal("fetch", replier(replies.baseUrl, replies.routes));

      const { parsed, isError } = await callThroughMcp(replies.baseUrl, entry.tool, entry.input);
      expect(isError, JSON.stringify(parsed)).toBe(false);

      expect(Object.keys(parsed).sort(), "the envelope's keys").toEqual(
        Object.keys(expected).sort(),
      );
      expect(parsed.data, "data").toEqual(expected.data);
      expect(parsed.success, "success").toEqual(expected.success);
      expect(parsed.count, "count").toEqual(expected.count);
      expect(parsed.pagination, "pagination").toEqual(expected.pagination);
      expect(typeof parsed.message, "message").toBe("string");
      expect((parsed.message as string).length).toBeGreaterThan(0);
      // Compared for the budgeted tools, because their sentence decides where the
      // page is cut; only required to exist for the rest. See the note at the top.
      if (MESSAGE_PARITY.has(entry.tool)) {
        expect(parsed.message, "message").toEqual(expected.message);
      }
    },
  );
});
