import { describe, it, expect, vi } from "vitest";
import { Readable, Writable } from "node:stream";
import { runInNewContext } from "node:vm";
import { z } from "zod";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import type { Transport } from "@modelcontextprotocol/sdk/shared/transport.js";
import type { JSONRPCMessage } from "@modelcontextprotocol/sdk/types.js";
import {
  keysTheSdkDrops,
  LaxArgumentsTransport,
  laxenCall,
  refuseDroppedKeys,
} from "../src/lax-transport";
import { buildServer } from "../src/server";

const shapes = new Map([["list_history_ids", { limit: z.number().int().optional() }]]);

/** A transport that records what it was asked to do and can be driven from the test. */
function fakeInner() {
  const inner: Transport & { sent: JSONRPCMessage[]; started: number; closed: number; session?: string } = {
    sent: [],
    started: 0,
    closed: 0,
    start: async () => void inner.started++,
    send: async (message: JSONRPCMessage) => void inner.sent.push(message),
    close: async () => void inner.closed++,
    get sessionId() {
      return inner.session;
    },
  };
  return inner;
}

const call = (args: unknown): JSONRPCMessage =>
  ({
    jsonrpc: "2.0",
    id: 1,
    method: "tools/call",
    params: { name: "list_history_ids", arguments: args },
  }) as unknown as JSONRPCMessage;

describe("the transport that decodes arguments", () => {
  it("passes an inner error to the callback the server installed", async () => {
    const inner = fakeInner();
    const wrapper = new LaxArgumentsTransport(inner, shapes);
    const seen: Error[] = [];
    wrapper.onerror = (error) => seen.push(error);
    await wrapper.start();
    inner.onerror?.(new Error("stdin exploded"));
    expect(seen.map((e) => e.message)).toEqual(["stdin exploded"]);
  });

  it("passes an inner close to the callback the server installed", async () => {
    const inner = fakeInner();
    const wrapper = new LaxArgumentsTransport(inner, shapes);
    const closed = vi.fn();
    wrapper.onclose = closed;
    await wrapper.start();
    inner.onclose?.();
    expect(closed).toHaveBeenCalledOnce();
  });

  /**
   * A host may have put its own callbacks on the transport before handing it over -- an HTTP
   * one drops its session on close -- and the SDK chains rather than replaces when it
   * installs its own. This does the same, or that cleanup silently stops running.
   */
  it("keeps callbacks that were already on the inner transport", async () => {
    const inner = fakeInner();
    const order: string[] = [];
    inner.onclose = () => order.push("host close");
    inner.onerror = () => order.push("host error");
    inner.onmessage = () => order.push("host message");
    const wrapper = new LaxArgumentsTransport(inner, shapes);
    wrapper.onclose = () => order.push("server close");
    wrapper.onerror = () => order.push("server error");
    wrapper.onmessage = () => order.push("server message");
    await wrapper.start();

    inner.onmessage?.(call({ limit: "5" }));
    inner.onerror?.(new Error("boom"));
    inner.onclose?.();

    expect(order).toEqual([
      "host message",
      "server message",
      "host error",
      "server error",
      "host close",
      "server close",
    ]);
  });

  it("gives the host's own handler the message as it arrived, not the decoded one", async () => {
    const inner = fakeInner();
    const seen: unknown[] = [];
    inner.onmessage = (message) => seen.push((message as never as { params: { arguments: unknown } }).params.arguments);
    const wrapper = new LaxArgumentsTransport(inner, shapes);
    wrapper.onmessage = (message) => seen.push((message as never as { params: { arguments: unknown } }).params.arguments);
    await wrapper.start();
    inner.onmessage?.(call({ limit: "5" }));
    expect(seen).toEqual([{ limit: "5" }, { limit: 5 }]);
  });

  it("is the inner transport for everything but that one field", async () => {
    const inner = fakeInner();
    const wrapper = new LaxArgumentsTransport(inner, shapes);
    await wrapper.start();
    await wrapper.send({ jsonrpc: "2.0", id: 2, result: {} } as JSONRPCMessage);
    await wrapper.close();
    expect([inner.started, inner.sent.length, inner.closed]).toEqual([1, 1, 1]);
    // Read live rather than copied: an HTTP transport assigns one during initialize.
    expect(wrapper.sessionId).toBeUndefined();
    inner.session = "abc123";
    expect(wrapper.sessionId).toBe("abc123");
    wrapper.setProtocolVersion("2025-06-18");
  });

  it("decodes a tool call's arguments and leaves every other message alone", async () => {
    const inner = fakeInner();
    const wrapper = new LaxArgumentsTransport(inner, shapes);
    const seen: JSONRPCMessage[] = [];
    wrapper.onmessage = (message) => seen.push(message);
    await wrapper.start();
    inner.onmessage?.(call({ limit: "5" }));
    inner.onmessage?.({ jsonrpc: "2.0", id: 9, method: "tools/list" } as JSONRPCMessage);
    expect((seen[0] as never as { params: { arguments: unknown } }).params.arguments).toEqual({ limit: 5 });
    expect(seen[1]).toEqual({ jsonrpc: "2.0", id: 9, method: "tools/list" });
  });

  /**
   * A malformed argument container is a malformed call, and the SDK already had an answer for
   * it. Spreading one into an object would turn it into a call with no arguments at all, which
   * would run the listing on its defaults instead of refusing. What goes on is the copy, because
   * the fields around it were read here -- but the value in the `arguments` field is the value
   * that was read, handed over as it was, so the SDK refuses the thing the caller sent.
   */
  it.each([[[]], ["nope"], [5], [true]])("leaves arguments: %j exactly as it found them", (args) => {
    const message = call(args);
    const read = laxenCall(message, shapes);
    expect(read).toEqual(message);
    expect((read as never as { params: { arguments: unknown } }).params.arguments).toBe(args);
  });

  /**
   * A decoder that throws takes the whole request with it: the caller gets no reply at all
   * where it used to get a JSON-RPC error. Asking a hostile `params.name` for a string is
   * the easy way to do that.
   */
  it.each([
    [{ name: { toString: 0 }, arguments: {} }],
    [{ name: 42, arguments: {} }],
    [{ name: null, arguments: {} }],
    [{ arguments: { limit: "5" } }],
    ["not params at all"],
    [null],
  ])("never throws on malformed params: %j", (params) => {
    const message = { jsonrpc: "2.0", id: 1, method: "tools/call", params } as unknown as JSONRPCMessage;
    expect(() => laxenCall(message, shapes)).not.toThrow();
    expect(laxenCall(message, shapes)).toEqual(message);
  });

  it("leaves a call for a tool it does not know", () => {
    const message = { ...call({ limit: "5" }), params: { name: "not_a_tool", arguments: { limit: "5" } } } as JSONRPCMessage;
    expect(laxenCall(message, shapes)).toEqual(message);
  });

  /**
   * Whatever it does decode, it hands on a COPY -- and that is the point, not a detail. Every
   * fallback used to hand back the object these fields were read off, so the SDK read them a
   * second time; an `arguments` accessor answering `undefined` first and `{"__proto__":{}}` after
   * got a dropped key past the refusal that way. The message that arrived leaves here only when
   * nothing here has read it.
   */
  it("hands on a copy of every message it has read, and the message itself otherwise", () => {
    for (const args of [{ limit: "5" }, {}, [], "nope", undefined]) {
      const message = call(args);
      expect(laxenCall(message, shapes)).not.toBe(message);
    }
    for (const method of ["tools/list", "initialize", "notifications/cancelled"]) {
      const message = { ...call({}), method } as unknown as JSONRPCMessage;
      expect(laxenCall(message, shapes)).toBe(message);
    }
  });

  /**
   * The eight fields the SDK's schemas declare, read the way it reads them.
   *
   * A zod object reads a declared field by property access, so a non-enumerable own `id` and an
   * inherited `_meta` or `task` are fields the SDK sees -- and a rebuild that only spread the own
   * enumerable keys lost all three. A message that arrived without a field still goes on without
   * it, because absence and `undefined` are different messages to that same schema.
   */
  it("carries the fields the SDK declares, however the caller attached them", () => {
    const meta = { progressToken: 7 };
    const task = { ttl: 5 };
    const params = Object.assign(Object.create({ _meta: meta, task }) as object, {
      name: "list_history_ids",
      arguments: { limit: "5" },
    });
    const message = { jsonrpc: "2.0", method: "tools/call", params } as unknown as JSONRPCMessage;
    Object.defineProperty(message, "id", { value: 1, enumerable: false });

    const read = laxenCall(message, shapes) as never as {
      jsonrpc: string;
      id: unknown;
      params: { name: string; arguments: unknown; _meta: unknown; task: unknown };
    };

    expect(read.jsonrpc).toBe("2.0");
    expect(read.id).toBe(1);
    expect(read.params.name).toBe("list_history_ids");
    expect(read.params.arguments).toEqual({ limit: 5 });
    expect(read.params._meta).toBe(meta);
    expect(read.params.task).toBe(task);
    // Own and enumerable on the copy, whatever they were on the caller's object: that is how the
    // strict envelope and the params object read them back.
    expect(Object.keys(read).sort()).toEqual(["id", "jsonrpc", "method", "params"]);
    expect(Object.keys(read.params).sort()).toEqual(["_meta", "arguments", "name", "task"]);
  });

  it("leaves out a field the caller did not send, rather than sending undefined for it", () => {
    const params = { name: "list_history_ids", arguments: {} };
    const notification = { jsonrpc: "2.0", method: "tools/call", params } as unknown as JSONRPCMessage;
    const read = laxenCall(notification, shapes) as never as { params: object };
    expect("id" in read).toBe(false);
    expect(Object.keys(read.params).sort()).toEqual(["arguments", "name"]);
    // And one written as undefined stays written as undefined, since that is a different message.
    const explicit = { ...notification, id: undefined } as unknown as JSONRPCMessage;
    expect("id" in laxenCall(explicit, shapes)).toBe(true);
  });
});

describe("the keys the SDK's argument parser drops", () => {
  /**
   * Read off the object, so what is refused is whatever that parser will throw away rather
   * than a list of names kept by hand. The two it throws away are `__proto__`, which it skips
   * by name so the plain object it builds into cannot have its prototype replaced, and any own
   * property that is not enumerable.
   */
  it("is __proto__, from an object written as JSON text", () => {
    expect(keysTheSdkDrops(JSON.parse('{"a":1,"__proto__":{}}'))).toEqual(["__proto__"]);
  });

  it("is a non-enumerable own property, which no JSON text can express", () => {
    const args = { a: 1 };
    Object.defineProperty(args, "historyId", { value: "h1", enumerable: false });
    expect(keysTheSdkDrops(args)).toEqual(["historyId"]);
  });

  /**
   * The names that look dangerous and are not: both are ordinary keys to the record parser and
   * to zod's object parser, so both reach the tool's own closed schema and are refused there.
   */
  it("is not constructor, prototype, or any ordinary key", () => {
    expect(keysTheSdkDrops(JSON.parse('{"constructor":1,"prototype":2,"nope":3,"limit":4}'))).toEqual([]);
  });

  it("does not look inside a value", () => {
    expect(keysTheSdkDrops(JSON.parse('{"representation":{"__proto__":{}}}'))).toEqual([]);
  });

  /** A refusal needs somebody to send it to; a notification asked for no reply. */
  it("is not answered for a message with no id", () => {
    const params = { name: "list_history_ids", arguments: JSON.parse('{"__proto__":{}}') };
    const notification = { jsonrpc: "2.0", method: "tools/call", params } as unknown as JSONRPCMessage;
    expect(refuseDroppedKeys(notification, shapes)).toBeUndefined();
    expect(refuseDroppedKeys({ ...notification, id: 1 } as JSONRPCMessage, shapes)).toBeDefined();
  });

  it("is not answered for a tool this server does not have", () => {
    const params = { name: "not_a_tool", arguments: JSON.parse('{"__proto__":{}}') };
    const message = { jsonrpc: "2.0", id: 1, method: "tools/call", params } as unknown as JSONRPCMessage;
    expect(refuseDroppedKeys(message, shapes)).toBeUndefined();
  });

  /**
   * The containers the SDK will read, rather than the ones this file would have preferred.
   *
   * Its request schema asks `params` for nothing more than `typeof`, and its record parser asks
   * `arguments` the question `isPlainObject` now asks -- so a class instance carrying a real
   * `name`, and an argument list another realm parsed, are both calls it runs and drops the key
   * out of. A guard of our own in front of either one meant the call went through unanswered.
   */
  it("answers a call whose params or arguments only an in-process peer could build", () => {
    const dropped = '{"__proto__":{}}';
    class Params {
      name = "list_history_ids";
      arguments = JSON.parse(dropped) as Record<string, unknown>;
    }
    const message = (params: unknown) =>
      ({ jsonrpc: "2.0", id: 1, method: "tools/call", params }) as unknown as JSONRPCMessage;
    expect(refuseDroppedKeys(message(new Params()), shapes)).toBeDefined();
    expect(
      refuseDroppedKeys(
        message({
          name: "list_history_ids",
          arguments: runInNewContext("JSON.parse(text)", { text: dropped }),
        }),
        shapes,
      ),
    ).toBeDefined();
  });

  it("never throws on a message it cannot make sense of", () => {
    for (const params of [null, "nope", 5, { name: { toString: 0 } }, { arguments: {} }]) {
      const message = { jsonrpc: "2.0", id: 1, method: "tools/call", params } as unknown as JSONRPCMessage;
      expect(() => refuseDroppedKeys(message, shapes)).not.toThrow();
      expect(refuseDroppedKeys(message, shapes)).toBeUndefined();
    }
  });
});

describe("a server behind it", () => {
  it("hears the peer close, and stops believing it is connected", async () => {
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const server = buildServer({ baseUrl: "https://g.example", apiKey: "K" });
    const client = new Client({ name: "close-check", version: "0" });
    const closed = vi.fn();
    server.server.onclose = closed;
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
    expect(server.server.transport).toBeDefined();

    await client.close();
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(closed).toHaveBeenCalled();
    // The Protocol clears its transport and rejects anything still in flight when it runs
    // its close path; a wrapper that swallowed onclose left it here believing otherwise.
    expect(server.server.transport).toBeUndefined();
    await expect(server.server.ping()).rejects.toThrow(/[Nn]ot connected/);
  });

  /**
   * `connect` refuses a second transport before it calls `start`, so the wiring has to wait
   * for `start`: a wrapper built for the rejected attempt must not re-point the inner
   * transport's callbacks at itself, or the live connection goes deaf and the host's close
   * handling never runs.
   */
  it("keeps working after a second connect is refused", async () => {
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const server = buildServer({ baseUrl: "https://g.example", apiKey: "K" });
    const client = new Client({ name: "double-connect", version: "0" });
    const closed = vi.fn();
    server.server.onclose = closed;
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);

    await expect(server.connect(serverTransport)).rejects.toThrow(/Already connected/);

    // The first connection still answers.
    const { tools } = await client.listTools();
    expect(tools.length).toBeGreaterThan(0);

    // And still hears the peer go away.
    await client.close();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(closed).toHaveBeenCalled();
    expect(server.server.transport).toBeUndefined();
  });

  it("leaves the transport alone when a second connect is refused", async () => {
    // The behavioural half above survives either way once the callbacks chain, so this is the
    // invariant itself: a connect that was refused has touched nothing. Wiring in start() is
    // what makes it true -- connect throws before it ever calls start.
    const inner = fakeInner();
    const server = buildServer({ baseUrl: "https://g.example", apiKey: "K" });
    await server.connect(inner);
    const wired = [inner.onmessage, inner.onerror, inner.onclose];

    await expect(server.connect(inner)).rejects.toThrow(/Already connected/);

    expect([inner.onmessage, inner.onerror, inner.onclose]).toEqual(wired);
  });

  it("hears a malformed line on a stdio stream", async () => {
    const stdin = new Readable({ read() {} });
    const written: string[] = [];
    const stdout = new Writable({
      write(chunk, _enc, done) {
        written.push(String(chunk));
        done();
      },
    });
    const server = buildServer({ baseUrl: "https://g.example", apiKey: "K" });
    const errors: Error[] = [];
    server.server.onerror = (error) => errors.push(error);
    await server.connect(new StdioServerTransport(stdin, stdout));

    stdin.push("{ this is not json }\n");
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(errors, "a parse error on stdin never reached the server").toHaveLength(1);
    expect(errors[0]!.message).toMatch(/JSON|token/i);
    await server.close();
  });

  /**
   * The `__proto__` refusal over a transport that really does parse JSON text, which is where
   * the key comes from: over stdio the SDK reads the line, `JSON.parse` makes `__proto__` an
   * ordinary own property of the argument list, and the answer written back down stdout has to
   * be the refusal rather than the result of a created history.
   */
  it("refuses __proto__ off a stdio line, and asks Galaxy for nothing", async () => {
    const fetched: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation((async (input: unknown) => {
      fetched.push(typeof input === "string" ? input : (input as Request).url);
      return new Response("{}", { status: 200, headers: { "content-type": "application/json" } });
    }) as unknown as typeof fetch);
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
      '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":' +
        '{"name":"create_history","arguments":{"history_name":"x","__proto__":{}}}}\n',
    );
    for (let tick = 0; tick < 20 && written.length === 0; tick++) {
      await new Promise((resolve) => setTimeout(resolve, 5));
    }

    const reply = JSON.parse(written.join("")) as {
      id: number;
      result: { isError: boolean; content: Array<{ text: string }> };
    };
    expect(reply.id).toBe(7);
    expect(reply.result.isError).toBe(true);
    expect(reply.result.content[0]!.text).toContain('Unrecognized parameter: \\"__proto__\\"');
    expect(fetched).toEqual([]);
    await server.close();
    vi.restoreAllMocks();
  });
});
