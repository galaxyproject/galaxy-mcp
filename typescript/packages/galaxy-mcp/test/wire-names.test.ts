import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { runInNewContext } from "node:vm";
import { z, type ZodType } from "zod";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { allOperations } from "@galaxyproject/galaxy-ops";
import { buildServer } from "../src/server";
import { toOperationInput, toSnakeCase, wireShape } from "../src/wire-names";
import { loadManifest, loadRegistry } from "./parity/surfaces";

/** Every request one test made: the URL it asked for and what it sent. */
let sent: Array<{ url: string; body: string }> = [];

beforeEach(() => {
  sent = [];
  vi.spyOn(globalThis, "fetch").mockImplementation((async (input: unknown, init?: RequestInit) => {
    const url =
      typeof input === "string" ? input : input instanceof URL ? input.href : (input as Request).url;
    // openapi-fetch hands `fetch` a Request, so what was posted is on that rather than in init.
    const body =
      input instanceof Request
        ? await input.clone().text()
        : typeof init?.body === "string"
          ? init.body
          : "";
    sent.push({ url, body });
    const answer = url.includes("/api/version")
      ? { version_major: "26.1", version_minor: "26.1.1" }
      : url.includes("/api/users/")
      ? { id: "u1", email: "me@example.invalid", username: "me" }
      : url.includes("/api/histories/")
        ? { id: "h1", name: "one" }
        : url.includes("/api/tools/")
          ? { id: "t1", name: "one" }
          : url.includes("/api/unprivileged_tools")
            ? { id: "u1", uuid: "u-1" }
            : [{ id: "p1", title: "one" }];
    return new Response(JSON.stringify(answer), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }) as unknown as typeof fetch);
});

afterEach(() => vi.restoreAllMocks());

async function withClient<T>(use: (client: Client) => Promise<T>): Promise<T> {
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  const server = buildServer({ baseUrl: "https://g.example", apiKey: "K" });
  const client = new Client({ name: "wire-names", version: "0" });
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
  try {
    return await use(client);
  } finally {
    await client.close();
    await server.close();
  }
}

/**
 * One `tools/call` written by hand and sent down the wire, with no SDK client in between.
 * The client builds a request from a typed object; a peer that is not this SDK writes
 * whatever it likes, including the shapes a typed object cannot express.
 */
async function rawCall(params: Record<string, unknown>) {
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  const server = buildServer({ baseUrl: "https://g.example", apiKey: "K" });
  await server.connect(serverTransport);
  const replies: Array<Record<string, unknown>> = [];
  let arrived: () => void = () => {};
  const firstReply = new Promise<void>((resolve) => (arrived = resolve));
  clientTransport.onmessage = (message) => {
    replies.push(message as Record<string, unknown>);
    arrived();
  };
  await clientTransport.start();
  await clientTransport.send({ jsonrpc: "2.0", id: 1, method: "tools/call", params } as never);
  // Wait for the answer rather than for a fixed number of ticks, because how many ticks a call
  // takes depends on how many requests it makes -- and give up after a moment, because a
  // message the server is right to ignore gets no answer at all.
  await Promise.race([firstReply, new Promise((resolve) => setTimeout(resolve, 2_000))]);
  await server.close();
  return replies[0] as {
    error?: unknown;
    result?: { isError?: boolean; content?: Array<{ text?: string }> };
  };
}

/** The text a raw reply carries, the way a client reads it out of the result. */
const rawText = (reply: { result?: { content?: Array<{ text?: string }> } }): string =>
  (reply.result?.content ?? []).map((c) => c.text ?? "").join("");

async function call(name: string, args?: Record<string, unknown>) {
  return withClient(async (client) => {
    // `arguments` is left off the request entirely when there are none to send, which is
    // what a client that knows the tool takes nothing does.
    const request = args === undefined ? { name } : { name, arguments: args as never };
    const result = (await client.callTool(request)) as {
      isError?: boolean;
      content?: Array<{ text?: string }>;
    };
    const text = (result.content ?? []).map((c) => c.text ?? "").join("");
    return { isError: result.isError === true, text };
  });
}

/**
 * A refusal message as it appears in what the client is handed: the SDK dumps the zod issues
 * as JSON into the error text, so the quotes around a parameter name arrive escaped.
 */
const escaped = (message: string): string => JSON.stringify(message).slice(1, -1);

const asked = (route: string): string => {
  const hit = sent.find((r) => r.url.includes(route));
  expect(hit, `nothing asked for ${route}`).toBeDefined();
  return hit!.url;
};

async function advertised(): Promise<Map<string, { properties: string[]; required: string[] }>> {
  return withClient(async (client) => {
    const { tools } = await client.listTools();
    return new Map(
      tools.map((t) => {
        const schema = t.inputSchema as { properties?: object; required?: string[] };
        return [
          t.name,
          { properties: Object.keys(schema.properties ?? {}), required: schema.required ?? [] },
        ];
      }),
    );
  });
}

describe("the parameter names this surface advertises", () => {
  it("is every op's input under the Python spelling", async () => {
    const tools = await advertised();
    for (const op of allOperations) {
      const expected = Object.keys(op.input).map(toSnakeCase);
      expect(tools.get(op.name)?.properties, op.name).toEqual(expected);
    }
  });

  it("says which of them are required under the same names", async () => {
    const tools = await advertised();
    for (const op of allOperations) {
      // Requiredness read off the op rather than off the advertisement, so a rename that
      // moved a parameter between the two lists would show up here rather than agree with
      // itself: a schema that takes nothing at all is what `.optional()` and `.default()`
      // both leave behind, and it is what the advertised `required` list leaves out.
      const required = Object.entries(op.input)
        .filter(([, schema]) => !(schema as ZodType).safeParse(undefined).success)
        .map(([key]) => toSnakeCase(key));
      expect([...(tools.get(op.name)?.required ?? [])].sort(), op.name).toEqual(required.sort());
    }
  });

  /**
   * The spelling itself, against the server it has to match. The parity check compares the
   * two surfaces parameter by parameter and no longer folds the two spellings together, so
   * this is the same contract read from the other end: a name the Python server declares and
   * this one does not is either a rename gone wrong or a difference somebody wrote down.
   */
  it("is what the Python server calls the same parameter", async () => {
    const tools = await advertised();
    const owed = new Set(
      loadRegistry()
        .divergences.filter((d) => d.kind === "missing-ts-param")
        .map((d) => `${d.tool}.${d.param}`),
    );
    const missing: string[] = [];
    for (const tool of loadManifest().tools) {
      const here = tools.get(tool.name);
      if (!here) continue; // a tool only that server has; the parity check lists those
      const properties = (tool.inputSchema as { properties?: object }).properties ?? {};
      for (const name of Object.keys(properties)) {
        if (here.properties.includes(name) || owed.has(`${tool.name}.${name}`)) continue;
        missing.push(`${tool.name}.${name}`);
      }
    }
    expect(missing).toEqual([]);
  });

  it("has no camelCase left in it", async () => {
    for (const [name, tool] of await advertised()) {
      expect(
        tool.properties.filter((p) => p !== toSnakeCase(p)),
        name,
      ).toEqual([]);
    }
  });

  /**
   * Not just the names -- the prose around them. A tool whose help says "pass sectionId" is
   * telling a client to make a call this server refuses, and a client has no way to know the
   * sentence is written in another surface's dialect. Read off the ops rather than from a
   * list kept by hand, so a description written tomorrow is held to the same rule.
   */
  it("says no parameter name in a spelling this surface refuses", async () => {
    const renamed = [
      ...new Set(
        allOperations.flatMap((op) => Object.keys(op.input).filter((k) => toSnakeCase(k) !== k)),
      ),
    ];
    expect(renamed.length).toBe(29);
    await withClient(async (client) => {
      const { tools } = await client.listTools();
      // The whole advertisement, not only the descriptions: a title, an enum or a default
      // carrying the old spelling would mislead a reader just as well.
      const said = (text: string) => renamed.filter((key) => new RegExp(`\\b${key}\\b`).test(text));
      for (const tool of tools) expect(said(JSON.stringify(tool)), tool.name).toEqual([]);
      expect(said(client.getInstructions() ?? "")).toEqual([]);
    });
  });

  /** The four sentences that told a caller to send a name this server now refuses. */
  it("tells a caller to pass the name it takes", async () => {
    const tools = await withClient(async (client) => (await client.listTools()).tools);
    const find = (name: string) => tools.find((t) => t.name === name)!;
    const property = (name: string, param: string) =>
      ((find(name).inputSchema as { properties: Record<string, { description?: string }> })
        .properties[param]?.description ?? "");

    expect(find("get_tool_panel").description).toContain("Pass section_id to list one section");
    expect(property("get_tool_panel", "limit")).toContain("when section_id is absent");
    expect(property("get_tool_panel", "offset")).toContain("to a section_id call");
    expect(find("list_pages").description).toContain("Pass history_id to list only");
    expect(find("create_page").description).toContain("With history_id it is a notebook");
    expect(property("invoke_workflow", "history_name")).toContain("ignored if history_id");
  });

  /**
   * The respelling is a rename of the op's OWN parameters and nothing else. Every one of
   * these words is a parameter somewhere, and every one of them is also ordinary English in
   * a sentence that has nothing to do with that parameter.
   */
  it("leaves the single-word parameters, and the English around them, alone", async () => {
    const tools = await withClient(async (client) => (await client.listTools()).tools);
    const description = (name: string) => tools.find((t) => t.name === name)!.description ?? "";
    expect(description("get_histories")).toContain("(id, name, counts)");
    expect(description("get_workflow_details")).toContain("(name, steps, inputs)");
    expect(description("update_history")).toContain("(name, annotation, tags, deleted, published)");
    expect(description("run_tool")).toContain("poll the jobs with get_job_details");
  });

  /**
   * Closed, the way all 46 Python tools are closed. This is the schema half of refusing an
   * unknown key; the call half is further down.
   */
  it("says it takes nothing else, on every tool", async () => {
    return withClient(async (client) => {
      const { tools } = await client.listTools();
      expect(tools.length).toBe(allOperations.length);
      for (const tool of tools) {
        const schema = tool.inputSchema as { additionalProperties?: unknown };
        expect(schema.additionalProperties, tool.name).toBe(false);
      }
    });
  });

  it("refuses a shape whose parameters collide on one wire name", () => {
    expect(() =>
      wireShape({ name: "example", input: { toolId: z.string(), tool_id: z.string() } }),
    ).toThrow(/both spelled "tool_id"/);
  });
});

describe("a validated call, on its way to the op", () => {
  const { toInput } = wireShape({
    name: "example",
    input: { historyId: z.string(), inputs: z.record(z.string(), z.unknown()).optional() },
  });

  it("arrives under the names the op reads", () => {
    expect(toOperationInput({ history_id: "h1" }, toInput)).toEqual({ historyId: "h1" });
  });

  it("leaves what is inside an object-valued parameter alone", () => {
    const inputs = { myInput: { src: "hda", id: "abc" }, "1|repeatValue": 2 };
    expect(toOperationInput({ history_id: "h1", inputs }, toInput).inputs).toEqual(inputs);
  });

  it("carries nothing the shape does not declare", () => {
    expect(toOperationInput({ history_id: "h1", historyId: "h2" }, toInput)).toEqual({
      historyId: "h1",
    });
  });

  it("leaves a parameter that was not given unset, rather than undefined", () => {
    expect(Object.keys(toOperationInput({ history_id: "h1" }, toInput))).toEqual(["historyId"]);
  });
});

describe("a tool call under the Python names", () => {
  it("takes the wire name and runs the op with it", async () => {
    const { isError, text } = await call("get_history_details", { history_id: "h1" });
    expect(isError, text).toBe(false);
    expect(asked("/api/histories/h1")).toBeDefined();
  });

  /**
   * The breaking half. A camelCase spelling is not a second name for the same parameter --
   * it is a name this surface does not take, and a tool here refuses a name it does not
   * take, as every Python tool does.
   */
  it("refuses the library spelling of a required parameter", async () => {
    const { isError, text } = await call("get_history_details", { historyId: "h1" });
    expect(isError).toBe(true);
    expect(text).toContain("Input validation error");
    expect(text).toContain(escaped('Unrecognized parameter: "historyId" -- did you mean "history_id"?'));
    expect(sent.some((r) => r.url.includes("/api/histories/"))).toBe(false);
  });

  /**
   * The dangerous one, and the reason this is refused rather than dropped: an optional
   * parameter under the old spelling used to be stripped as an unknown key and the tool ran
   * on its defaults -- an upload into the wrong history, a filter that matched everything.
   */
  it("refuses the library spelling of an optional one, rather than running without it", async () => {
    const { isError, text } = await call("get_tool_details", { tool_id: "t1", ioDetails: true });
    expect(isError).toBe(true);
    expect(text).toContain(escaped('Unrecognized parameter: "ioDetails" -- did you mean "io_details"?'));
    expect(sent.some((r) => r.url.includes("/api/tools/"))).toBe(false);
  });

  it("refuses a name that is nobody's spelling of anything, without guessing", async () => {
    const { isError, text } = await call("get_tool_details", { tool_id: "t1", nope: 1 });
    expect(isError).toBe(true);
    expect(text).toContain(escaped('Unrecognized parameter: "nope"'));
    expect(text).not.toContain("did you mean");
  });

  it("refuses an unknown key on a tool that takes no parameters at all", async () => {
    const { isError, text } = await call("get_user", { limit: 1 });
    expect(isError).toBe(true);
    expect(text).toContain(escaped('Unrecognized parameter: "limit"'));
    expect(sent.some((r) => r.url.includes("/api/users"))).toBe(false);
  });

  it("names every unknown key, not just the first", async () => {
    const { isError, text } = await call("list_pages", { historyId: "h1", nope: 1 });
    expect(isError).toBe(true);
    expect(text).toContain(
      escaped('Unrecognized parameters: "historyId" -- did you mean "history_id"?, "nope"'),
    );
  });

  /**
   * The one key a closed schema cannot refuse, because it never gets to see it.
   *
   * It has to arrive as JSON text: the object literal `{"__proto__": {}}` sets the prototype
   * instead of making a property, so a test written that way would be testing nothing at all.
   * `JSON.parse` defines it as an ordinary own property -- which is what a client on the far end
   * of a real connection sends, and what the SDK's argument parser then quietly drops, leaving
   * the tool to run as though the caller had never written it.
   */
  it("refuses a key the SDK's argument parser drops, in the words the schema would have used", async () => {
    const args = JSON.parse('{"history_name":"x","__proto__":{}}') as Record<string, unknown>;
    expect(Object.hasOwn(args, "__proto__")).toBe(true);
    const dropped = await rawCall({ name: "create_history", arguments: args });
    expect(dropped.result?.isError, rawText(dropped)).toBe(true);
    expect(rawText(dropped)).toContain(escaped('Unrecognized parameter: "__proto__"'));
    expect(sent).toEqual([]);

    // The same call with an ordinary unknown key, refused by the closed schema through the SDK.
    // Nothing about the refusal above may differ from that one but the name inside it: same
    // prefix, same error code, same issue code, same JSON.
    const ordinary = await rawCall({
      name: "create_history",
      arguments: { history_name: "x", nope: {} },
    });
    expect(ordinary.result?.isError, rawText(ordinary)).toBe(true);
    expect(rawText(dropped)).toBe(rawText(ordinary).split("nope").join("__proto__"));
    expect(sent).toEqual([]);
  });

  /**
   * The same refusal for the two containers only an in-process caller can deliver.
   *
   * `InMemoryTransport` hands the server the object the peer built, so a host that runs untrusted
   * text in a `vm` context can pass an `arguments` whose prototype is that context's
   * `Object.prototype`, and a peer can pass a `params` that is a class instance. The SDK takes
   * both and drops the key out of both: its record parser asks whether the value's constructor
   * looks like `Object` rather than whose `Object.prototype` it has, and its request schema asks
   * `params` for nothing more than `typeof`. So a refusal that asked either question its own way
   * never saw these calls, and `create_history` ran with a key the caller wrote and nobody kept.
   */
  it("refuses that key on a container only an in-process caller can deliver", async () => {
    const asJsonText = '{"history_name":"x","__proto__":{}}';
    const sameRealm = await rawCall({
      name: "create_history",
      arguments: JSON.parse(asJsonText) as Record<string, unknown>,
    });
    expect(sameRealm.result?.isError, rawText(sameRealm)).toBe(true);
    expect(sent).toEqual([]);

    // An argument list this realm did not parse.
    sent = [];
    const elsewhere = runInNewContext("JSON.parse(text)", { text: asJsonText }) as Record<string, unknown>;
    expect(Object.getPrototypeOf(elsewhere)).not.toBe(Object.prototype);
    const foreign = await rawCall({ name: "create_history", arguments: elsewhere });
    expect(rawText(foreign)).toBe(rawText(sameRealm));
    expect(sent).toEqual([]);

    // And a `params` that is not a plain object at all, carrying the same argument list.
    sent = [];
    class Params {
      name = "create_history";
      arguments = JSON.parse(asJsonText) as Record<string, unknown>;
    }
    const instance = await rawCall(new Params() as unknown as Record<string, unknown>);
    expect(rawText(instance)).toBe(rawText(sameRealm));
    expect(sent).toEqual([]);
  });

  /**
   * The two names the review asked about next to that one. Both are ordinary keys to the SDK's
   * parser and to zod's, so both reach the tool's own schema and are refused there -- nothing in
   * the transport needs to know about them, and `keysTheSdkDrops` says so in its own test.
   */
  it("leaves constructor and prototype to the schema, which refuses them like any other name", async () => {
    for (const key of ["constructor", "prototype"]) {
      sent = [];
      const args = JSON.parse(`{"history_name":"x","${key}":{}}`) as Record<string, unknown>;
      const reply = await rawCall({ name: "create_history", arguments: args });
      expect(reply.result?.isError, key).toBe(true);
      expect(rawText(reply)).toContain(escaped(`Unrecognized parameter: "${key}"`));
      expect(sent).toEqual([]);
    }
  });

  /**
   * Inside an object-valued parameter the same name is the caller's own data. The rename stops
   * at the top level and so does the refusal: a representation on its way to Galaxy arrives with
   * whatever the caller wrote in it.
   */
  it("lets __proto__ inside an object-valued parameter through untouched", async () => {
    const args = JSON.parse(
      '{"representation":{"class":"GalaxyUserTool","id":"t","version":"0.1","name":"t",' +
        '"shell_command":"echo hi","container":"python:3.12-slim",' +
        '"nested":{"__proto__":{"deep":1}}}}',
    ) as Record<string, unknown>;
    const reply = await rawCall({ name: "create_user_tool", arguments: args });
    expect(reply.result?.isError, rawText(reply)).toBe(false);
    const posted = sent.find((r) => r.url.includes("/api/unprivileged_tools"));
    expect(posted, JSON.stringify(sent.map((s) => s.url))).toBeDefined();
    expect(posted!.body).toContain('"nested":{"__proto__":{"deep":1}}');
  });

  it("takes a call that sends no arguments at all, for a tool that needs none", async () => {
    const { isError, text } = await call("get_user", undefined);
    expect(isError, text).toBe(false);
    expect(asked("/api/users/current")).toBeDefined();
  });

  it("still refuses a call that sends no arguments to a tool that needs one", async () => {
    const { isError, text } = await call("get_history_details", undefined);
    expect(isError).toBe(true);
    expect(text).toContain("history_id");
    expect(sent.some((r) => r.url.includes("/api/histories/"))).toBe(false);
  });

  it("decodes a renamed argument the way the other surface decodes it", async () => {
    // The lax decoding reads the shape by the name a caller wrote, so a renamed parameter
    // gets the same "yes" -> true it got under the old spelling.
    const { isError, text } = await call("list_pages", { show_published: "yes" });
    expect(isError, text).toBe(false);
    expect(asked("/api/pages")).toContain("show_published=true");
  });

  /**
   * The same call from a peer that is not this SDK's client, which is where a missing
   * `arguments` really comes from: the client above builds the request from a typed object,
   * but MCP makes the field optional and another implementation may simply leave it out.
   */
  it("takes a raw call with no arguments field at all", async () => {
    const reply = await rawCall({ name: "get_user" });
    expect(reply.error).toBeUndefined();
    expect(reply.result?.isError).toBe(false);
    expect(JSON.stringify(reply)).toContain("me@example.invalid");
  });

  /**
   * Present, and undefined. Only a transport that carries objects can deliver this -- JSON
   * has no way to write it -- and it is not the same message as the one above: the peer put
   * the field in and what it put there is not an argument list. Reading the value rather
   * than asking whether the field is there would make it `{}` and run the tool, which for
   * `get_user` means an unasked-for request going out over a real connection.
   */
  it("refuses a raw call whose arguments field is there but undefined", async () => {
    const reply = await rawCall({ name: "get_user", arguments: undefined });
    expect(reply.result?.isError).toBe(true);
    expect(JSON.stringify(reply)).toContain("expected object, received undefined");
    expect(sent).toEqual([]);
  });

  it("hands an object-valued parameter over with its own keys untouched", async () => {
    const representation = {
      class: "GalaxyUserTool",
      id: "t",
      version: "0.1",
      name: "t",
      shell_command: "echo hi",
      container: "python:3.12-slim",
      // Not a parameter of the tool -- a key inside the user's own document, which is on its
      // way to Galaxy and is none of this surface's business.
      someNestedKey: { andAnother: 1 },
    };
    const { isError, text } = await call("create_user_tool", { representation });
    expect(isError, text).toBe(false);
    const posted = sent.find((r) => r.url.includes("/api/unprivileged_tools"));
    expect(JSON.parse(posted!.body)).toEqual({ src: "representation", representation });
  });
});
