import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { Readable, Writable } from "node:stream";
import { runInNewContext } from "node:vm";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { buildServer } from "../src/server";

/**
 * What a caller writes inside an object-valued parameter, and what Galaxy is sent.
 *
 * Both this package's README and the changelog promise that the keys inside such a parameter
 * -- a workflow's `inputs`, a tool's parameters, a user tool's `representation` -- are the
 * caller's own data and reach Galaxy exactly as written. One key made that false: zod's record
 * parser, which every one of these parameters used to be, copies the object into a fresh `{}`
 * and skips `__proto__` while copying. The call then succeeded and posted a document the
 * caller had not written, which is the one thing a surface must not do quietly. The Python
 * server takes the same parameters as `dict[str, Any]` and pydantic keeps the key.
 *
 * `jsonObject()` in `@galaxyproject/galaxy-ops` is the fix: the same JSON Schema, and the record
 * still doing the parsing, with the key it drops by name defined back onto the copy it made.
 * Every one of these tests fails on the commit before it, at the depth that matters -- the TOP
 * level of the parameter's own value, which is one layer above the nested case
 * `wire-names.test.ts` already covers. Three of them go the other way, at values only an
 * in-process caller can deliver: they say what the copy and the record's own key validation are
 * still there for. A fourth such value fails without the fix rather than with it -- a
 * representation that deletes one of its own keys while the record is reading it, where the
 * answer owed is about the document that arrived.
 */

/** Every request one test made: the URL it asked for and the body it sent. */
let sent: Array<{ url: string; body: string }> = [];

/** The answers Galaxy gives, by the piece of URL that picks one. Anything else is a 404. */
function mockGalaxy(answers: Array<[string, unknown]>): void {
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
    const answer = answers.find(([route]) => url.includes(route));
    return new Response(JSON.stringify(answer ? answer[1] : { err_msg: "no" }), {
      status: answer ? 200 : 404,
      headers: { "content-type": "application/json" },
    });
  }) as unknown as typeof fetch);
}

const VERSION: [string, unknown] = ["/api/version", { version_major: "26.1", version_minor: "26.1.1" }];

beforeEach(() => {
  sent = [];
});

afterEach(() => vi.restoreAllMocks());

/**
 * One `tools/call` written by hand, with its arguments built by `JSON.parse` rather than
 * from an object literal: `{"__proto__": {}}` written as a literal sets the prototype instead
 * of making a property, so a test written that way would be testing nothing at all.
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
  await Promise.race([firstReply, new Promise((resolve) => setTimeout(resolve, 5_000))]);
  await server.close();
  return replies[0] as { result?: { isError?: boolean; content?: Array<{ text?: string }> } };
}

const rawText = (reply: { result?: { content?: Array<{ text?: string }> } }): string =>
  (reply.result?.content ?? []).map((c) => c.text ?? "").join("");

/**
 * The body of the one request that went to `route`.
 *
 * An exact path first, because a tool that reads a schema before it runs asks
 * `/api/tools/<id>` on the way to posting `/api/tools`, and a substring match would hand back
 * the GET. `/invocations` is a path suffix rather than a whole path, so a substring match is
 * still the fallback.
 */
function postedTo(route: string): string {
  const hit =
    sent.find((r) => new URL(r.url).pathname === route) ?? sent.find((r) => r.url.includes(route));
  expect(hit, `nothing was sent to ${route}; asked for ${JSON.stringify(sent.map((s) => s.url))}`).toBeDefined();
  return hit!.body;
}

/** The pollution-flavoured key, written the way a client on a real connection sends it. */
const PROTO = '"__proto__":{"direct":1}';

interface Case {
  /** The tool, and the parameter whose own top level carries the key. */
  tool: string;
  param: string;
  /** The `arguments` object, as JSON text. */
  args: string;
  /** What Galaxy answers, and the route whose body is the proof. */
  answers: Array<[string, unknown]>;
  route: string;
  /** The body that request must carry, byte for byte. */
  body: string;
}

const CASES: Case[] = [
  {
    tool: "create_user_tool",
    param: "representation",
    args:
      '{"representation":{"class":"GalaxyUserTool","id":"t","version":"0.1","name":"t",' +
      `"shell_command":"echo hi","container":"python:3.12-slim",${PROTO}}}`,
    answers: [VERSION, ["/api/unprivileged_tools", { id: "u1", uuid: "u-1" }]],
    route: "/api/unprivileged_tools",
    body:
      '{"src":"representation","representation":{"class":"GalaxyUserTool","id":"t","version":"0.1",' +
      `"name":"t","shell_command":"echo hi","container":"python:3.12-slim",${PROTO}}}`,
  },
  {
    tool: "run_tool",
    param: "inputs",
    args: `{"tool_id":"fastqc/0.74","history_id":"h1","inputs":{${PROTO},"input_file":{"src":"hda","id":"d1"}}}`,
    answers: [VERSION, ["/api/tools", { outputs: [], jobs: [] }]],
    route: "/api/tools",
    body:
      '{"history_id":"h1","tool_id":"fastqc/0.74","input_format":"legacy",' +
      `"inputs":{${PROTO},"input_file":{"src":"hda","id":"d1"}}}`,
  },
  {
    tool: "run_user_tool",
    param: "inputs",
    args: `{"history_id":"h1","tool_uuid":"u-1","inputs":{${PROTO}}}`,
    answers: [
      VERSION,
      ["/api/unprivileged_tools/u-1", { tool_id: "t1", representation: { version: "0.2" } }],
      ["/api/tools", { outputs: [], jobs: [] }],
    ],
    route: "/api/tools",
    body: `{"history_id":"h1","tool_uuid":"u-1","tool_version":"0.2","inputs":{${PROTO}},"input_format":"legacy"}`,
  },
  {
    tool: "invoke_workflow",
    param: "inputs",
    // Non-empty inputs run the preflight first. Nothing answers the workflow and datatype
    // routes here, so it gives up and the invocation goes out -- which is the path under test.
    args: `{"workflow_id":"w1","inputs":{${PROTO}}}`,
    answers: [VERSION, ["/invocations", { id: "i1", state: "new" }]],
    route: "/invocations",
    body: `{"inputs":{${PROTO}},"inputs_by":"step_index","parameters":{},"parameters_normalized":false}`,
  },
  {
    tool: "invoke_workflow",
    param: "params",
    args: `{"workflow_id":"w1","params":{${PROTO}}}`,
    answers: [VERSION, ["/invocations", { id: "i1", state: "new" }]],
    route: "/invocations",
    body: `{"inputs":{},"inputs_by":"step_index","parameters":{${PROTO}},"parameters_normalized":false}`,
  },
];

describe("a key at the top level of an object-valued parameter", () => {
  for (const c of CASES) {
    it(`reaches Galaxy from ${c.tool}.${c.param}`, async () => {
      mockGalaxy(c.answers);
      const args = JSON.parse(c.args) as Record<string, unknown>;
      const reply = await rawCall({ name: c.tool, arguments: args });
      expect(reply.result?.isError, rawText(reply)).toBe(false);
      expect(postedTo(c.route)).toBe(c.body);
    });
  }

  /**
   * The other direction of the same promise: a value an in-process caller can deliver and a
   * client cannot.
   *
   * `InMemoryTransport` carries live objects, so `arguments.representation` need not have been
   * parsed by this realm's `JSON.parse` -- a host that runs untrusted text in a `vm` context
   * hands over an object whose prototype is that context's `Object.prototype`. zod's record
   * called it plain and the call reached Galaxy; a check that asked for this realm's prototype
   * by identity refused it with "expected record, received Object". Which is why the check is
   * zod's own predicate and not a restatement of it.
   */
  it("takes a representation that another realm parsed", async () => {
    mockGalaxy([VERSION, ["/api/unprivileged_tools", { id: "u1", uuid: "u-1" }]]);
    const { representation } = JSON.parse(CASES[0]!.args) as { representation: unknown };
    const text = JSON.stringify(representation);
    const elsewhere = runInNewContext("JSON.parse(text)", { text }) as Record<string, unknown>;
    // The probe is only a probe if the object really is foreign.
    expect(Object.getPrototypeOf(elsewhere)).not.toBe(Object.prototype);
    const reply = await rawCall({ name: "create_user_tool", arguments: { representation: elsewhere } });
    expect(reply.result?.isError, rawText(reply)).toBe(false);
    expect(postedTo("/api/unprivileged_tools")).toBe(CASES[0]!.body);
  });

  /**
   * The other thing a live object can carry, and the reason the value is still copied: a
   * serialisation hook of its own.
   *
   * `JSON.stringify` asks the value for a `toJSON` and posts whatever that returns, and the
   * property answering does not have to be enumerable -- so a representation whose fields all
   * validate can serialise as a string. zod's record left such a property behind when it copied,
   * and a version of this schema that handed the arriving object straight back did not, which
   * made the document Galaxy was posted the hook's choice rather than the caller's.
   */
  it("posts the fields of a representation that carries a hidden toJSON", async () => {
    mockGalaxy([VERSION, ["/api/unprivileged_tools", { id: "u1", uuid: "u-1" }]]);
    const { representation } = JSON.parse(CASES[0]!.args) as { representation: object };
    Object.defineProperty(representation, "toJSON", { value: () => "wrong-document" });
    // The probe is only a probe if the hook really would have taken over.
    expect(JSON.stringify(representation)).toBe('"wrong-document"');
    const reply = await rawCall({ name: "create_user_tool", arguments: { representation } });
    expect(reply.result?.isError, rawText(reply)).toBe(false);
    expect(postedTo("/api/unprivileged_tools")).toBe(CASES[0]!.body);
  });

  /**
   * A representation that rewrites itself while it is being read, which is the other thing a
   * live object can do.
   *
   * The record reads each key once, in the order it finds them, so `note` is copied before the
   * getter on `zap` runs and deleting it afterwards cannot unsay that -- the record's answer is
   * about the document that arrived. Asking the input for its key order again once the record had
   * finished lost `note` out of the body: a key the caller wrote, and this surface's one promise
   * about the inside of an object-valued parameter is that it does not lose one of those quietly.
   */
  it("posts a representation that deletes one of its own keys while being read", async () => {
    mockGalaxy([VERSION, ["/api/unprivileged_tools", { id: "u1", uuid: "u-1" }]]);
    const fields =
      '"class":"GalaxyUserTool","id":"t","version":"0.1","name":"t","shell_command":"echo hi",' +
      `"container":"python:3.12-slim",${PROTO},"note":"keep","zap":0`;
    const representation = JSON.parse(`{${fields}}`) as Record<string, unknown>;
    Object.defineProperty(representation, "zap", {
      get: () => {
        delete representation.note;
        return 1;
      },
      enumerable: true,
      configurable: true,
    });
    const reply = await rawCall({ name: "create_user_tool", arguments: { representation } });
    expect(reply.result?.isError, rawText(reply)).toBe(false);
    // Byte for byte the record's copy with the caller's `__proto__` back in it: every key they
    // wrote, in the order they wrote it, with the getter's own value where the getter was.
    expect(postedTo("/api/unprivileged_tools")).toBe(
      `{"src":"representation","representation":{${fields.replace('"zap":0', '"zap":1')}}}`,
    );
  });

  /**
   * And a key a record refused rather than dropped: a symbol.
   *
   * `Reflect.ownKeys` finds it, `z.string()` says it is not a key of this record, and the call is
   * refused with that issue. A schema that took the value instead would run `create_user_tool`
   * with a key that vanishes at serialisation -- the quiet answer again, in the other direction.
   */
  it("refuses a representation with a symbol key, in the record's words", async () => {
    mockGalaxy([VERSION]);
    const { representation } = JSON.parse(CASES[0]!.args) as { representation: object };
    // Enumerable, because a key the record cannot see is a key it never had to refuse.
    Object.defineProperty(representation, Symbol("unexpected"), { value: 123, enumerable: true });
    const reply = await rawCall({ name: "create_user_tool", arguments: { representation } });
    expect(reply.result?.isError, rawText(reply)).toBe(true);
    expect(rawText(reply)).toContain("Invalid key in record");
    expect(rawText(reply)).toContain('"invalid_key"');
    expect(sent.filter((r) => r.url.includes("/api/unprivileged_tools"))).toEqual([]);
  });

  /**
   * The same thing over a transport that really does parse JSON text, end to end: the line
   * goes into stdin, the reply comes back off stdout, and what Galaxy was posted in between
   * carries the caller's document with nothing taken out of it.
   */
  it("survives a whole stdio round trip", async () => {
    mockGalaxy([VERSION, ["/api/unprivileged_tools", { id: "u1", uuid: "u-1" }]]);
    const stdin = new Readable({ read() {} });
    const written: string[] = [];
    const stdout = new Writable({
      write(chunk, _enc, done) {
        written.push(String(chunk));
        done();
      },
    });
    const server = buildServer({ baseUrl: "https://g.example", apiKey: "K" });
    await server.connect(new StdioServerTransport(stdin, stdout));

    stdin.push(
      '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"create_user_tool",' +
        `"arguments":${CASES[0]!.args}}}\n`,
    );
    for (let tick = 0; tick < 100 && written.length === 0; tick++) {
      await new Promise((resolve) => setTimeout(resolve, 5));
    }

    const reply = JSON.parse(written.join("")) as {
      id: number;
      result: { isError: boolean; content: Array<{ text: string }> };
    };
    expect(reply.id).toBe(7);
    expect(reply.result.isError, reply.result.content[0]?.text).toBe(false);
    expect(postedTo("/api/unprivileged_tools")).toBe(CASES[0]!.body);
    await server.close();
  });

  /**
   * The parameter is still checked, in the words it was checked in before: a `z.record` said
   * "expected record" and so does this, so the only thing that changed for a caller is the
   * key that used to disappear.
   */
  it("still refuses a parameter that is not an object at all", async () => {
    mockGalaxy([VERSION]);
    for (const [value, received] of [
      ['"nope"', "string"],
      ["5", "number"],
      ["[1]", "array"],
      ["null", "null"],
    ]) {
      sent = [];
      const args = JSON.parse(`{"representation":${value}}`) as Record<string, unknown>;
      const reply = await rawCall({ name: "create_user_tool", arguments: args });
      expect(reply.result?.isError, rawText(reply)).toBe(true);
      expect(rawText(reply)).toContain(`Invalid input: expected record, received ${received}`);
      expect(sent.filter((r) => r.url.includes("/api/unprivileged_tools"))).toEqual([]);
    }
  });

  it("still refuses a call that leaves the parameter out", async () => {
    mockGalaxy([VERSION]);
    const reply = await rawCall({ name: "create_user_tool", arguments: {} });
    expect(reply.result?.isError, rawText(reply)).toBe(true);
    expect(rawText(reply)).toContain("Invalid input: expected record, received undefined");
  });

  /**
   * And the schema a client reads is the one it read before, byte for byte -- which is what
   * lets the parity check go on comparing these five against the Python manifest. Written out
   * in full rather than compared to the zod that produced it, so that a change to how the
   * schema is built has to be typed out here to pass.
   */
  it("advertises exactly the schema a record advertised", async () => {
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const server = buildServer({ baseUrl: "https://g.example", apiKey: "K" });
    const client = new Client({ name: "json-object-params", version: "0" });
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
    const { tools } = await client.listTools();
    const schemaOf = (tool: string, param: string): string => {
      const found = tools.find((t) => t.name === tool);
      expect(found, tool).toBeDefined();
      const properties = found!.inputSchema.properties as Record<string, unknown>;
      return JSON.stringify(properties[param]);
    };
    const open = '"type":"object","propertyNames":{"type":"string"},"additionalProperties":{}';

    expect(schemaOf("create_user_tool", "representation")).toBe(
      `{${open},"description":"a GalaxyUserTool representation: {class:'GalaxyUserTool', id, ` +
        `version, name, shell_command, container:'<image>'}"}`,
    );
    expect(schemaOf("run_tool", "inputs")).toBe(
      `{${open},"description":"Tool input parameters in Galaxy's legacy format: dataset inputs as ` +
        `{\\"input_name\\": {\\"src\\": \\"hda\\", \\"id\\": \\"dataset_id\\"}}; a parameter inside a ` +
        `section, conditional or repeat as one flat key joined with '|', e.g. ` +
        `\\"reference_source|ref_file\\" -- not nested objects."}`,
    );
    expect(schemaOf("run_user_tool", "inputs")).toBe(
      `{${open},"description":"tool inputs; dataset refs as {src:'hda',id}"}`,
    );
    // The two that are an object OR the JSON string of one, and nullable with it.
    expect(schemaOf("invoke_workflow", "inputs")).toBe(
      '{"description":"Workflow inputs keyed by step_index. Each value is {src, id} for ' +
        'datasets/collections or a scalar for parameters. A JSON object string is accepted too.",' +
        `"anyOf":[{"anyOf":[{${open}},{"type":"string"}]},{"type":"null"}]}`,
    );
    expect(schemaOf("invoke_workflow", "params")).toBe(
      '{"description":"Legacy step parameter overrides (use inputs for formal inputs instead). ' +
        'A JSON object string is accepted too.",' +
        `"anyOf":[{"anyOf":[{${open}},{"type":"string"}]},{"type":"null"}]}`,
    );
    // And they are required exactly where they were.
    const required = (tool: string) =>
      (tools.find((t) => t.name === tool)!.inputSchema.required as string[]) ?? [];
    expect(required("create_user_tool")).toEqual(["representation"]);
    expect(required("run_tool")).toEqual(["tool_id", "history_id", "inputs"]);
    expect(required("run_user_tool")).toEqual(["history_id", "tool_uuid", "inputs"]);
    expect(required("invoke_workflow")).toEqual(["workflow_id"]);

    await client.close();
    await server.close();
  });
});
