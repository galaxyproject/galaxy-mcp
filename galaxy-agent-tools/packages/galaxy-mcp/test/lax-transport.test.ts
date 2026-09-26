import { describe, it, expect, vi } from "vitest";
import { Readable, Writable } from "node:stream";
import { z } from "zod";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import type { Transport } from "@modelcontextprotocol/sdk/shared/transport.js";
import type { JSONRPCMessage } from "@modelcontextprotocol/sdk/types.js";
import { LaxArgumentsTransport, laxenCall } from "../src/lax-transport";
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
   * would run the listing on its defaults instead of refusing.
   */
  it.each([[[]], ["nope"], [5], [true]])("leaves arguments: %j exactly as it found them", (args) => {
    const message = call(args);
    expect(laxenCall(message, shapes)).toBe(message);
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
    expect(laxenCall(message, shapes)).toBe(message);
  });

  it("leaves a call for a tool it does not know", () => {
    const message = { ...call({ limit: "5" }), params: { name: "not_a_tool", arguments: { limit: "5" } } } as JSONRPCMessage;
    expect(laxenCall(message, shapes)).toBe(message);
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
});
