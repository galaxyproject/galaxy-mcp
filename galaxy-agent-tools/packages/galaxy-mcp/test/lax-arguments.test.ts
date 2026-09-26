import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { buildServer } from "../src/server";

/**
 * What this surface accepts for an integer or a boolean argument, against what the Python
 * server accepts for the same one.
 *
 * FastMCP validates a tool call with `type_adapter.validate_python(arguments)` and never asks
 * pydantic for strict mode, so over there an integer parameter takes `"5"`, `"05"`, `"5.0"`,
 * `"1_000"`, `true` and `5.0`, and a boolean takes `"yes"`, `"off"`, `1` and `0`. Every row
 * below was measured against this repo's pydantic with `TypeAdapter(int | None)` and
 * `TypeAdapter(bool)` -- Python mode and JSON mode agree -- and is asserted here through a
 * real client, because the advertised schema still says `integer` and `boolean` and that is
 * deliberately not what decides this.
 */
const INTEGERS_ACCEPTED: Array<[unknown, number]> = [
  ["5", 5],
  [" 5 ", 5],
  ["+5", 5],
  ["-5", -5],
  ["05", 5],
  ["5.0", 5],
  ["1_000", 1000],
  [5.0, 5],
  [-0.0, 0],
  [true, 1],
  [false, 0],
  [1e2, 100],
];

/** The other half of the same table: what pydantic refuses, and so does this. */
const INTEGERS_REFUSED: unknown[] = ["5.5", 5.5, "1e2", "0x10", "", "abc", "٥", [1], { a: 1 }];

const BOOLEANS_ACCEPTED: Array<[unknown, boolean]> = [
  ["true", true],
  ["false", false],
  ["True", true],
  ["TRUE", true],
  ["Off", false],
  ["1", true],
  ["0", false],
  ["yes", true],
  ["no", false],
  ["on", true],
  ["off", false],
  [1, true],
  [0, false],
];

const BOOLEANS_REFUSED: unknown[] = [2, "", null, "maybe", [true]];

/** Every URL the server asked for during one test, so an accepted value can be read off the wire. */
let asked: string[] = [];

beforeEach(() => {
  asked = [];
  vi.spyOn(globalThis, "fetch").mockImplementation((async (input: unknown) => {
    const url =
      typeof input === "string" ? input : input instanceof URL ? input.href : (input as Request).url;
    asked.push(url);
    const body = url.includes("/api/version")
      ? { version_major: "26.1", version_minor: "26.1.1" }
      : url.includes("/api/workflows/")
        ? { id: "w1", name: "one" }
        : url.includes("/api/tools/")
          ? { id: "t1", name: "one" }
          : Array.from({ length: 10 }, (_, k) => ({ id: `h${k}`, name: `h${k}` }));
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }) as unknown as typeof fetch);
});

afterEach(() => vi.restoreAllMocks());

async function call(name: string, args: Record<string, unknown>) {
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  const server = buildServer({ baseUrl: "https://g.example", apiKey: "K" });
  const client = new Client({ name: "lax-check", version: "0" });
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
  try {
    const result = (await client.callTool({ name, arguments: args as never })) as {
      isError?: boolean;
      content?: Array<{ text?: string }>;
    };
    const text = (result.content ?? []).map((c) => c.text ?? "").join("");
    return { isError: result.isError === true, text };
  } finally {
    await client.close();
    await server.close();
  }
}

const query = (route: string): string => {
  const hit = asked.find((url) => url.includes(route));
  expect(hit, `nothing asked for ${route}`).toBeDefined();
  return hit!;
};

describe("an integer argument, decoded the way the other surface decodes it", () => {
  // The window rules run after the decoding, so this half of the table uses values that are
  // a legal page for this op: -5, 0 and 1000 decode fine and are then refused by the ceiling
  // and the floor, which is the op's business rather than the decoder's.
  it.each(INTEGERS_ACCEPTED.filter(([, want]) => want >= 1 && want <= 500))(
    "takes %j as %i, and pages with it",
    async (given, want) => {
      const { isError, text } = await call("list_history_ids", { limit: given as never });
      expect(isError, text).toBe(false);
      // The converted value, visible in what came back: ten histories, this many returned.
      const page = JSON.parse(text) as { data: { items: unknown[]; pagination: { limit: number } } };
      expect(page.data.pagination.limit).toBe(want);
      expect(page.data.items).toHaveLength(Math.min(want, 10));
    },
  );

  it("decodes a window the op then refuses, and says which rule refused it", async () => {
    // Decoding and validating are different jobs: "0" is a number here and not a page there.
    const { isError, text } = await call("list_history_ids", { limit: "0" as never });
    expect(isError).toBe(true);
    expect(text).toContain("limit must be at least 1 (got 0)");
    expect(text).toContain('"errorKind":"validation"');
  });

  it.each(INTEGERS_ACCEPTED)("puts %j on the wire as %i", async (given, want) => {
    const { isError, text } = await call("list_pages", { offset: given as never });
    expect(isError, text).toBe(false);
    expect(query("/api/pages")).toContain(`offset=${want}`);
  });

  it.each(INTEGERS_REFUSED)("refuses %j, as pydantic does", async (given) => {
    const { isError, text } = await call("list_history_ids", { limit: given as never });
    expect(isError).toBe(true);
    expect(text).toContain("Input validation error");
  });

  it("takes null for the one integer the other surface declares as nullable", async () => {
    const { isError, text } = await call("get_workflow_details", { workflowId: "w1", version: null });
    expect(isError, text).toBe(false);
    const { isError: refused } = await call("list_pages", { offset: null as never });
    expect(refused).toBe(true);
  });

  it("takes a converted version on the wire too", async () => {
    const { isError, text } = await call("get_workflow_details", { workflowId: "w1", version: "3" });
    expect(isError, text).toBe(false);
    expect(query("/api/workflows/w1")).toContain("version=3");
  });
});

/**
 * The whitespace half of the same table, through the same client. `trim()` is not pydantic's
 * whitespace in two places, and both are here: U+0085 is stripped there and not by `trim()`,
 * U+FEFF is stripped by `trim()` and refused there. One per group otherwise.
 */
describe("the whitespace around an argument", () => {
  it.each([
    [" ", "ascii"],
    ["\u0085", "next line"],
    ["\u00a0", "no-break space"],
    ["\u1680", "ogham"],
    ["\u2007", "figure space"],
    ["\u2028", "line separator"],
    ["\u3000", "ideographic space"],
  ])("strips %j (%s) and pages with what is left", async (space) => {
    const { isError, text } = await call("list_pages", { offset: `${space}7${space}` as never });
    expect(isError, text).toBe(false);
    expect(query("/api/pages")).toContain("offset=7");
  });

  it.each([
    ["\ufeff", "zero width no-break space"],
    ["\u200b", "zero width space"],
  ])("refuses %j (%s), as pydantic does", async (space) => {
    const { isError } = await call("list_pages", { offset: `${space}7${space}` as never });
    expect(isError).toBe(true);
  });
});

/**
 * The decoder runs on whatever a caller sent, before anything has refused it, so it has to
 * be linear in the length of that string. It was not: trimming with `^space+|space+$` made
 * the engine try the trailing branch from every position, and 64K spaces followed by a
 * non-digit -- an argument that was always going to be refused -- held the thread for about
 * two seconds. The bound here is generous enough not to flake on a loaded machine and still
 * two orders off the quadratic version.
 */
describe("an argument nobody should be able to make expensive", () => {
  it("refuses 64K spaces in well under the time the old trim took", async () => {
    const hostile = `5${" ".repeat(64_000)}x`;
    const started = performance.now();
    const { isError } = await call("list_history_ids", { limit: hostile as never });
    const elapsed = performance.now() - started;
    expect(isError).toBe(true);
    expect(elapsed, `took ${elapsed.toFixed(0)}ms`).toBeLessThan(200);
  });

  it("is linear in the length of the string, not quadratic", async () => {
    const time = async (spaces: number) => {
      const started = performance.now();
      await call("list_history_ids", { limit: `5${" ".repeat(spaces)}x` as never });
      return performance.now() - started;
    };
    await time(2_000);
    const small = await time(16_000);
    const large = await time(64_000);
    // Four times the input took four times as long when this was quadratic, so sixteen.
    expect(large, `16K took ${small.toFixed(0)}ms, 64K took ${large.toFixed(0)}ms`).toBeLessThan(
      Math.max(small, 1) * 8 + 50,
    );
  });
});

/**
 * A malformed argument container is a malformed call. Decoding one would spread it into an
 * empty object and run the listing on its defaults; the server's own refusal is the right
 * answer and is what these still get.
 *
 * The SDK's client will not send one -- it validates the request before it goes out -- so
 * these go down the wire raw, the way a peer that is not this SDK would send them. A Date, a
 * Map, a Set, a boxed number and a class instance are in the list because an
 * object-carrying transport really can deliver them: JSON text cannot, but this one passes
 * references, and `typeof` calls every one of them an object.
 */
describe("an arguments container that is not an object", () => {
  it.each([
    [[], "an array"],
    ["nope", "a string"],
    [5, "a number"],
    [true, "a boolean"],
    [new Date(0), "a Date"],
    [new Map([["limit", 5]]), "a Map"],
    [new Set([1]), "a Set"],
    // eslint-disable-next-line no-new-wrappers
    [new Number(5), "a boxed number"],
    [new (class Args { limit = 5 })(), "a class instance"],
  ] as Array<[unknown, string]>)("is refused, not decoded: %s", async (args) => {
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const server = buildServer({ baseUrl: "https://g.example", apiKey: "K" });
    await server.connect(serverTransport);
    const replies: Array<Record<string, unknown>> = [];
    clientTransport.onmessage = (message) => replies.push(message as Record<string, unknown>);
    await clientTransport.start();
    await clientTransport.send({
      jsonrpc: "2.0",
      id: 1,
      method: "tools/call",
      params: { name: "list_history_ids", arguments: args },
    } as never);
    await new Promise((resolve) => setTimeout(resolve, 0));
    await server.close();

    expect(JSON.stringify(replies[0])).toMatch(/error|isError/);
    expect(asked.some((url) => url.includes("/api/histories")), "it ran the listing anyway").toBe(false);
  });
});

describe("a boolean argument, decoded the way the other surface decodes it", () => {
  it.each(BOOLEANS_ACCEPTED)("takes %j as %j on get_tool_details", async (given, want) => {
    const { isError, text } = await call("get_tool_details", { toolId: "t1", ioDetails: given as never });
    expect(isError, text).toBe(false);
    expect(query("/api/tools/t1")).toContain(`io_details=${want}`);
  });

  it.each(BOOLEANS_ACCEPTED)("takes %j as %j on list_workflows", async (given, want) => {
    const { isError, text } = await call("list_workflows", { published: given as never });
    expect(isError, text).toBe(false);
    // bioblend sends the flag only when it is true, which is what a false decodes to here.
    expect(query("/api/workflows").includes("show_published=true")).toBe(want);
  });

  it.each(BOOLEANS_REFUSED)("refuses %j, as pydantic does", async (given) => {
    const { isError, text } = await call("get_tool_details", { toolId: "t1", ioDetails: given as never });
    expect(isError).toBe(true);
    expect(text).toContain("Input validation error");
  });
});
