import { isPlainObject, laxenLikePydantic } from "@galaxyproject/galaxy-ops";
import type { Transport, TransportSendOptions } from "@modelcontextprotocol/sdk/shared/transport.js";
import type { JSONRPCMessage, MessageExtraInfo } from "@modelcontextprotocol/sdk/types.js";
import type { ZodRawShape } from "zod";

/**
 * One inbound message, with a tool call's arguments decoded and everything else as it was.
 *
 * Only `tools/call`, only a tool this server registered, and only an `arguments` that is a
 * plain object -- decided by prototype, since a transport that carries objects rather than
 * JSON text can deliver a Date, a Map or a boxed number and `typeof` calls all of them
 * objects. An array, a string, a number, a boolean or any of those is a malformed call, and
 * it goes through untouched so the SDK refuses it exactly as it did before any of this.
 *
 * Nothing in here may throw. A message this cannot make sense of is a message for the SDK to
 * answer, and a decoder that threw would take the whole request with it -- the caller would
 * get no reply at all where it used to get a JSON-RPC error. `params.name` alone can do that:
 * a peer is free to send `{name: {toString: 0}}`, and asking that for a string is a TypeError.
 */
export function laxenCall<T extends JSONRPCMessage>(message: T, shapes: Map<string, ZodRawShape>): T {
  try {
    const call = message as unknown as { method?: string; params?: { name?: unknown; arguments?: unknown } };
    if (call.method !== "tools/call" || !isPlainObject(call.params)) return message;
    if (typeof call.params.name !== "string") return message;
    const shape = shapes.get(call.params.name);
    if (!shape || !isPlainObject(call.params.arguments)) return message;
    return {
      ...message,
      params: { ...call.params, arguments: laxenLikePydantic(shape, call.params.arguments) },
    } as T;
  } catch {
    return message;
  }
}

/**
 * A transport that decodes a tool call's arguments on the way in, and is otherwise the
 * transport it was given.
 *
 * The decoding has to happen here rather than in a tool's handler: the SDK validates
 * arguments against the advertised schema before it calls a handler, so by then a `"5"` has
 * already been refused. Nor can it live in the schema -- a `z.preprocess` or a pipe advertises
 * as `{"type":"object","properties":{}}` through this SDK, which throws away the contract this
 * whole branch exists to match. The other candidate was a `tools/call` handler on the
 * low-level server, but `setRequestHandler` replaces rather than wraps, and the SDK's own
 * handler is where tool lookup, input and output validation and task support live.
 *
 * What it does not touch: the message stream in either direction beyond that one field, the
 * order of anything, or the inner transport's lifecycle. `start`, `send` and `close` are the
 * inner transport's, and `sessionId` is read from it live rather than copied, because an
 * HTTP transport assigns one during initialize.
 */
export class LaxArgumentsTransport implements Transport {
  onmessage?: (message: JSONRPCMessage, extra?: MessageExtraInfo) => void;
  onerror?: (error: Error) => void;
  onclose?: () => void;

  private wired = false;

  constructor(
    private readonly inner: Transport,
    private readonly shapes: Map<string, ZodRawShape>,
  ) {}

  /**
   * Wiring happens here rather than in the constructor because `connect` refuses a second
   * transport before it ever calls this: wiring earlier would let a rejected second attempt
   * re-point the inner transport's callbacks at a wrapper nobody is connected to, and the
   * live connection would stop hearing messages and closes.
   */
  start(): Promise<void> {
    if (!this.wired) {
      this.wired = true;
      // Chain rather than replace, the way the SDK itself does when it installs its
      // callbacks: an HTTP host may already have put its own session cleanup on onclose
      // before handing the transport over, and that has to keep running. The pre-existing
      // handler is given the message as it arrived; decoding is this connection's business,
      // not theirs.
      const { onmessage, onerror, onclose } = this.inner;
      this.inner.onmessage = (message, extra) => {
        onmessage?.(message, extra);
        this.onmessage?.(laxenCall(message, this.shapes), extra);
      };
      this.inner.onerror = (error) => {
        onerror?.(error);
        this.onerror?.(error);
      };
      this.inner.onclose = () => {
        onclose?.();
        this.onclose?.();
      };
    }
    return this.inner.start();
  }

  send(message: JSONRPCMessage, options?: TransportSendOptions): Promise<void> {
    return this.inner.send(message, options);
  }

  close(): Promise<void> {
    return this.inner.close();
  }

  get sessionId(): string | undefined {
    return this.inner.sessionId;
  }

  setProtocolVersion(version: string): void {
    this.inner.setProtocolVersion?.(version);
  }
}
