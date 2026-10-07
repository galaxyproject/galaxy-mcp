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
 * * `message` deep-equal, byte for byte. The sentence is part of the contract for
 *   every tool: it is what an agent reads first, and for the nine tools whose page
 *   the output budget cuts it also decides where the cut falls, because the budget
 *   is measured on the whole envelope and a sentence eleven bytes longer on one
 *   side cuts one row fewer there.
 *
 * Key ORDER is not part of the contract. JSON objects are unordered, no client
 * depends on it, and neither side promises one.
 *
 * A case whose `outcome` is "failure" pins something else entirely, because a failed call
 * carries no envelope: the other server raises, FastMCP turns the exception into a tool
 * result with `isError` and one text block, and the fixture holds that result as the wire
 * carries it. Those cases are compared whole -- the blocks, their text byte for byte, and
 * the flag -- because the text IS the answer there and there is nothing else in it.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";
import { describe, it, expect, afterEach, vi } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { buildServer } from "../src/server";
// The IWC manifest and the container recommender are both memoised for the life of
// the process, as they are on the other side, so a case that serves its own manifest
// or its own tag list has to start from nothing remembered. Reached directly because
// they are test hooks and not part of the package's public surface.
import { __resetIwcCacheForTest } from "../../galaxy-ops/src/iwc-manifest";
import { __clearRecommendationCacheForTest } from "../../galaxy-ops/src/mulled";

const FIXTURES = new URL(
  "../../../../mcp-server-galaxy-py/tests/testdata/envelopes/",
  import.meta.url,
);

interface Route {
  method: string;
  path: string;
  query: Record<string, string>;
  /**
   * Keys the JSON request body has to carry, with these values; any others it carries are
   * ignored. Only on a route that has to prove an argument reached a write's payload.
   */
  json?: Record<string, unknown>;
  status: number;
  headers: Record<string, string>;
  /** The reply as a value, for a route whose exact bytes do not matter. */
  body?: unknown;
  /**
   * The reply as bytes, for a route whose do. A failure sentence quotes the body -- twice,
   * for a bioblend GET, once as a Python `bytes` repr -- so a route that feeds one has to
   * answer both sides with the same bytes and not merely with the same value.
   */
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

/** A failed call as the other server's wire carries it, plus the tool's own sentence. */
interface FailureCase {
  outcome: "failure";
  result: { content: Array<{ type: string; text: string }>; isError: true };
  sentence: string;
}

const readJson = <T>(name: string): T =>
  JSON.parse(readFileSync(fileURLToPath(new URL(name, FIXTURES)), "utf8")) as T;

const index = readJson<{ caseCount: number; cases: CaseEntry[] }>("index.json");

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
    const sent = routes.some((route) => route.json) ? await sentJson(input, init) : undefined;
    const target = routes
      .filter((route) => {
        if (route.method.toUpperCase() !== method) return false;
        const want = route.path.startsWith("http")
          ? new URL(route.path)
          : new URL(route.path, baseUrl);
        if (want.origin !== url.origin || want.pathname !== url.pathname) return false;
        if (!Object.entries(route.query).every(([k, v]) => url.searchParams.get(k) === v)) {
          return false;
        }
        return Object.entries(route.json ?? {}).every(
          ([k, v]) => sent !== undefined && isDeepStrictEqual(sent[k], v),
        );
      })
      .sort((a, b) => specificity(b) - specificity(a))[0];
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

/** How much of a request a route names: the most specific matching route wins. */
const specificity = (route: Route): number =>
  Object.keys(route.query).length + Object.keys(route.json ?? {}).length;

/**
 * The request's JSON body as an object, or undefined when there is none to read. openapi-fetch
 * hands over a Request, whose body is read from a clone so the reply path is left untouched.
 */
async function sentJson(input: unknown, init?: { body?: unknown }): Promise<Record<string, unknown> | undefined> {
  const text =
    input instanceof Request
      ? await input.clone().text()
      : typeof init?.body === "string"
        ? init.body
        : "";
  if (!text) return undefined;
  try {
    const parsed: unknown = JSON.parse(text);
    return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : undefined;
  } catch {
    return undefined;
  }
}

/** Call one tool the way a client does, and hand back the parsed text block. */
async function callThroughMcp(
  baseUrl: string,
  tool: string,
  args: Record<string, unknown>,
): Promise<{ content: Array<{ type: string; text?: string }>; isError: boolean }> {
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  const server = buildServer({ baseUrl, apiKey: "envelope-parity-not-a-key" });
  const client = new Client({ name: "envelope-parity", version: "0" });
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
  try {
    const result = (await client.callTool({ name: tool, arguments: args })) as {
      content: Array<{ type: string; text?: string }>;
      isError?: boolean;
    };
    return { content: result.content, isError: result.isError === true };
  } finally {
    await client.close();
    await server.close();
  }
}

afterEach(() => {
  vi.unstubAllGlobals();
  __resetIwcCacheForTest();
  __clearRecommendationCacheForTest();
});

const successes = index.cases.filter((c) => c.outcome === "success");
const failures = index.cases.filter((c) => c.outcome === "failure");

/** Point the runtime at this case's canned Galaxy, from nothing remembered. */
function arrange(replies: { baseUrl: string; routes: Route[] }): void {
  __resetIwcCacheForTest();
  __clearRecommendationCacheForTest();
  vi.stubGlobal("fetch", replier(replies.baseUrl, replies.routes));
}

describe("the MCP envelope is the Python server's", () => {
  it("has cases to check", () => {
    expect(index.cases.length).toBe(index.caseCount);
    expect(successes.length).toBeGreaterThan(40);
  });

  it.each(successes.map((c) => [`${c.tool} / ${c.case}`, c] as const))(
    "%s",
    async (_label, entry) => {
      const expected = readJson<Record<string, unknown>>(entry.envelope);
      const replies = readJson<{ baseUrl: string; routes: Route[] }>(entry.replies);
      arrange(replies);

      const { content, isError } = await callThroughMcp(replies.baseUrl, entry.tool, entry.input);
      const text = content.map((c) => c.text ?? "").join("");
      expect(isError, text).toBe(false);
      const parsed = JSON.parse(text) as Record<string, unknown>;

      expect(Object.keys(parsed).sort(), "the envelope's keys").toEqual(
        Object.keys(expected).sort(),
      );
      expect(parsed.data, "data").toEqual(expected.data);
      expect(parsed.success, "success").toEqual(expected.success);
      expect(parsed.count, "count").toEqual(expected.count);
      expect(parsed.pagination, "pagination").toEqual(expected.pagination);
      expect(typeof parsed.message, "message").toBe("string");
      expect((parsed.message as string).length).toBeGreaterThan(0);
      expect(parsed.message, "message").toEqual(expected.message);
    },
  );
});

describe("the MCP failure is the Python server's", () => {
  it("has cases to check", () => {
    expect(failures.length).toBeGreaterThan(10);
  });

  it.each(failures.map((c) => [`${c.tool} / ${c.case}`, c] as const))(
    "%s",
    async (_label, entry) => {
      const expected = readJson<FailureCase>(entry.envelope);
      const replies = readJson<{ baseUrl: string; routes: Route[] }>(entry.replies);
      arrange(replies);

      const { content, isError } = await callThroughMcp(replies.baseUrl, entry.tool, entry.input);
      const text = content.map((c) => c.text ?? "").join("");
      expect(isError, text).toBe(true);
      // One block, of type text, carrying the whole answer -- not an envelope with
      // success:false in it, which is what this surface used to send.
      expect(content.length, "content blocks").toBe(expected.result.content.length);
      expect(content[0]?.type, "the block's type").toBe(expected.result.content[0]?.type);
      expect(text, "the failure text").toEqual(expected.result.content[0]?.text);
      // And the tool's own sentence is in there, with FastMCP's wrapper in front of it.
      expect(text.endsWith(expected.sentence), "the tool's sentence").toBe(true);
    },
  );
});
