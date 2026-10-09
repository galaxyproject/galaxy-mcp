import { isPlainObject, laxenLikePydantic, materializeOnce } from "@galaxyproject/galaxy-ops";
import type { Transport, TransportSendOptions } from "@modelcontextprotocol/sdk/shared/transport.js";
import { ErrorCode, McpError } from "@modelcontextprotocol/sdk/types.js";
import type { JSONRPCMessage, MessageExtraInfo } from "@modelcontextprotocol/sdk/types.js";
import { ZodError, type ZodRawShape } from "zod";
import { unknownKeyMessage } from "./wire-names.js";

/** What one read of an inbound message yields. */
interface CallAsRead {
  /** The message to hand on -- the copy, unless nothing here read the one that arrived. */
  message: JSONRPCMessage;
  /** A response to send in the call's place, when a key arrived that the SDK's parser drops. */
  refusal?: JSONRPCMessage;
}

/**
 * The fields the SDK's own schemas read off a message, and off a `tools/call`'s params.
 *
 * `JSONRPCRequestSchema` is `{jsonrpc, id}` plus `RequestSchema`'s `{method, params}`, made
 * strict. `CallToolRequestParamsSchema` is `TaskAugmentedRequestParamsSchema` -- `_meta` and
 * `task` -- extended with `name` and `arguments`, and it strips what it does not declare rather
 * than refusing it. Neither has a catchall. A zod object reads a field it declares by PROPERTY
 * ACCESS, so inherited counts and non-enumerable counts, and the SDK reads `method`, `id`,
 * `params`, `params._meta` and `params.task` by hand around the parse besides.
 *
 * So these eight are read here, once each, the same way the SDK reads them, and put on the copy
 * as own enumerable data. Everything else on the copy is the own enumerable extras
 * `materializeOnce` read once -- which is what a spread carried, and what the strict envelope and
 * the stripping params object go on to do with them is their business, unchanged.
 */
const MESSAGE_FIELDS = ["jsonrpc", "id", "method", "params"] as const;
const PARAMS_FIELDS = ["name", "arguments", "_meta", "task"] as const;

/**
 * A field as read: the one value it gave, and whether it was there to give one.
 *
 * Absence and `undefined` are different messages to the SDK. A message with no `id` is a
 * notification it answers nothing to; one with an own `id: undefined` is a request with a bad id,
 * and the strict envelope turns it down. So a field that was not there stays not there on the
 * copy. The value is read first and the key asked about only when that came back `undefined`, so
 * an ordinary field costs one property access and no more -- and `in` runs no getter, which is
 * why asking it is not a second read of anything.
 */
interface Field {
  readonly present: boolean;
  readonly value: unknown;
}

function readField(from: object, key: string): Field {
  const value = (from as Record<string, unknown>)[key];
  return { value, present: value !== undefined || key in from };
}

/**
 * The field as read, put on the copy as own enumerable data.
 *
 * Defined rather than assigned: the copy keeps the caller's prototype, and a setter up there is
 * not this surface's to trigger.
 */
function carry(copy: Record<string, unknown>, key: string, field: Field): void {
  if (!field.present) return;
  Object.defineProperty(copy, key, {
    value: field.value,
    writable: true,
    enumerable: true,
    configurable: true,
  });
}

/**
 * One inbound message, read once: a tool call's arguments decoded, a dropped key refused, and
 * everything else as it was.
 *
 * READ ONCE is the rule this whole file is built on, and it is a rule because a transport that
 * carries objects rather than JSON text hands over a LIVE object. A peer on an `InMemoryTransport`
 * can pass a `params` whose `arguments` getter answers differently on its third read, or a Proxy
 * whose `ownKeys` reports one key set and then another. This surface used to read such a value
 * twice -- once to inspect it, and then the SDK read it again to parse it -- so what was inspected
 * and what ran were two different argument lists, and a key this promises to refuse got through.
 * The fix is not a better inspection: every field is read here exactly once, into inert data
 * (`materializeOnce`), and the message handed on carries those copies. The SDK's parse, the
 * dropped-key refusal and the lax decode all read the same copy, so no second read exists for a
 * hostile object to answer differently.
 *
 * Which means ONCE THIS HAS READ ANYTHING, ONLY THE COPY LEAVES. Every path out below -- nothing
 * to decode, a name that is not a name, a tool this server does not have, an argument list that
 * is not one, an absent `arguments`, and the refusal too -- hands on the rebuilt message, because
 * handing back the object these fields were read off is handing the SDK a second read. That is
 * what a `params` whose `arguments` answered `undefined` and then `{"__proto__":{}}` used to walk
 * through: the fallback forwarded the original, the SDK asked again, and the third answer ran
 * with the key dropped. The one exception is above the first read: a message whose `method` is
 * not `tools/call` goes on exactly as it came, since nothing was promised about it and nothing
 * here has looked at it.
 *
 * Fields are looked up in the shape the tool advertises, keyed by the names a caller writes
 * -- the same lookup the SDK is about to do, so a value is decoded for the schema that will
 * judge it and a name this surface does not declare is left for that judgement.
 *
 * Only `tools/call`, only a tool this server registered, and only an `arguments` the SDK's own
 * parser will take as a record -- `isPlainObject` is that parser's predicate, because a
 * transport that carries objects rather than JSON text can deliver a Date, a Map or a boxed
 * number and `typeof` calls all of them objects. An array, a string, a number, a boolean or any
 * of those is a malformed call, and it is carried across as it was read so the SDK refuses it
 * exactly as it did before any of this.
 *
 * Nothing in here may throw. A message this cannot make sense of is a message for the SDK to
 * answer, and a decoder that threw would take the whole request with it -- the caller would
 * get no reply at all where it used to get a JSON-RPC error. `params.name` alone can do that:
 * a peer is free to send `{name: {toString: 0}}`, and asking that for a string is a TypeError.
 * A throwing trap is the one case where the message that arrived is handed on after something
 * was read off it -- there is no copy to hand on instead, and an object that throws when it is
 * read throws for the SDK too, which is what `8a6f67c` did with one.
 */
function readCall(message: JSONRPCMessage, shapes: Map<string, ZodRawShape>): CallAsRead {
  try {
    // A message has to be an object for any of this: writing the fields read below onto an array
    // or a primitive would be writing onto the caller's own object rather than onto a copy. The
    // SDK has no answer for either one, before this or after it.
    if (typeof message !== "object" || message === null || Array.isArray(message)) return { message };
    const envelope = message as unknown as Record<string, unknown>;
    // The method is read once and the value read is the one handed on: a second read could
    // disagree with this one, and the SDK would dispatch a call this had decoded as another.
    const method = envelope.method;
    if (method !== "tools/call") return { message };
    const jsonrpc = readField(envelope, "jsonrpc");
    const id = readField(envelope, "id");
    const paramsField = readField(envelope, "params");
    const params = paramsField.value;
    /** The envelope rebuilt out of what was read, around the params given. */
    const asMessage = (nextParams: Field): JSONRPCMessage => {
      const next = materializeOnce(message, MESSAGE_FIELDS) as Record<string, unknown>;
      carry(next, "jsonrpc", jsonrpc);
      carry(next, "id", id);
      carry(next, "method", { present: true, value: method });
      carry(next, "params", nextParams);
      return next as unknown as JSONRPCMessage;
    };
    // The SDK's own gate for `params`: its request schema parses that field with `z.object`,
    // which asks `util.isObject` -- typeof, minus arrays -- rather than plain. So a class
    // instance carrying a real `name` is a call it runs, and a call this has to read the same
    // way it will. Anything else the SDK refuses as it always did, off the value read here.
    if (typeof params !== "object" || params === null || Array.isArray(params)) {
      return { message: asMessage(paramsField) };
    }
    // By property access, and inherited and non-enumerable both count, because that is how the
    // SDK reads a field it declares. `Object.assign(Object.create({name: "get_user"}), {arguments:
    // {}})` is a call it runs, and a rebuild that only spread the own enumerable keys answered
    // "expected string at params.name"; a `task` hidden the same way is a call it refuses, and
    // that rebuild ran it.
    const nameField = readField(params, "name");
    const argsField = readField(params, "arguments");
    const meta = readField(params, "_meta");
    const task = readField(params, "task");
    /** The whole message rebuilt, with the argument list given in place of the one read. */
    const withArguments = (args: Field): JSONRPCMessage => {
      const nextParams = materializeOnce(params, PARAMS_FIELDS) as Record<string, unknown>;
      carry(nextParams, "name", nameField);
      carry(nextParams, "arguments", args);
      carry(nextParams, "_meta", meta);
      carry(nextParams, "task", task);
      return asMessage({ present: true, value: nextParams });
    };
    /** The call as it was read, decoded by nothing: every path that decodes nothing ends here. */
    const asRead = (): CallAsRead => ({ message: withArguments(argsField) });
    const given = (args: unknown): Field => ({ present: true, value: args });

    const name = nameField.value;
    if (typeof name !== "string") return asRead();
    const shape = shapes.get(name);
    if (shape === undefined) return asRead();
    const args = argsField.value;
    if (args === undefined) {
      // A call that sends no arguments at all is a call with no arguments. MCP makes the field
      // optional, FastMCP reads a missing one as an empty mapping, and every tool here whose
      // parameters are all optional -- get_user takes none at all -- can be called that way on
      // the other server. Without this the SDK validates `undefined` against the tool's object
      // and answers "expected object, received undefined". A field that is present but not a
      // plain object is a different thing: that is a malformed call, and it is handed on as read
      // so the SDK refuses it exactly as it did before any of this.
      //
      // Absence is asked as absence: a transport that carries objects rather than JSON text can
      // deliver `{name, arguments: undefined}`, where the peer said something about the field and
      // what it said is not an argument list. Reading the value would have turned that into an
      // empty one and run the tool.
      if (argsField.present) return asRead();
      return { message: withArguments(given({})) };
    }
    // The one read of the caller's argument list, and the question of whether it IS one asked of
    // what came back rather than of the object it came from. The copy carries the original's
    // prototype, so the answer is the same one the SDK's record parser reaches, and the
    // `constructor` behind it is read once -- asked of the original first, it was two reads, and
    // a `constructor` answering `Object` and then `Date` sat in between them.
    const inert = materializeOnce(args);
    if (!isPlainObject(inert)) return { message: withArguments(given(inert)) };
    // Everything below reads this copy: the keys the SDK will drop are counted off it, the lax
    // decoding copies it, and the SDK parses what that produced -- three answers about one
    // object, read once.
    const dropped = keysTheSdkDrops(inert);
    // Without an id there is nobody to answer: a peer that sent this as a notification asked
    // for no reply, and inventing one would put a response on the wire for no request. The call
    // goes on with the key still in it, and the SDK drops it as it would have anyway. The id is
    // the one read of the id: the refusal quotes what was read, and so does the copy.
    if (dropped.length > 0 && id.value !== undefined && id.value !== null) {
      return {
        message: withArguments(given(inert)),
        refusal: droppedKeyRefusal(id.value, name, shape, dropped),
      };
    }
    return { message: withArguments(given(laxenLikePydantic(shape, inert))) };
  } catch {
    return { message };
  }
}

/**
 * One inbound message with a tool call's arguments decoded, or the message it was handed.
 *
 * The decoding, the gates and the reason nothing here may throw are all `readCall`'s; this is
 * the half of its answer that goes on to the server.
 */
export function laxenCall<T extends JSONRPCMessage>(message: T, shapes: Map<string, ZodRawShape>): T {
  return readCall(message, shapes).message as T;
}

/**
 * The keys of an argument list that the SDK's own parser throws away before any tool sees it.
 *
 * `tools/call` arguments are parsed as `z.record(z.string(), z.unknown())`, and zod's record
 * parser skips two kinds of key while copying the object it hands on: `__proto__`, by name and
 * unconditionally, so that the plain `{}` it builds into cannot have its prototype replaced by
 * an assignment; and any own property that is not enumerable. Closing the tools' schemas does
 * not help, because zod's object parser skips `__proto__` in exactly the same way -- so the key
 * is neither kept nor refused, and the tool runs as if the caller had not written it. A key
 * that arrived and disappeared is the one thing this surface must not do quietly now that the
 * names have changed, so it is refused here, which is the last place it can still be seen.
 *
 * Read off the object rather than by name alone, so the answer is "whatever the parser will
 * drop" rather than a list that can fall out of date. `constructor` and `prototype` are
 * ordinary keys to both parsers and are refused by the tool's own schema, like any other name
 * a tool does not declare -- there is a test that says so.
 *
 * The object it is asked about is the inert copy the SDK will go on to parse, never the caller's
 * own -- which is why `materializeOnce` carries a non-enumerable key into that copy by name: it
 * is a key that arrived, this is where it is seen, and the parsers skip it there as they would
 * have skipped it on the object it came from.
 *
 * Only the top level of the argument list. What is inside an object-valued parameter is the
 * caller's own data on its way to Galaxy and is none of this surface's business.
 */
export function keysTheSdkDrops(args: Record<string, unknown>): string[] {
  return Object.getOwnPropertyNames(args).filter(
    (key) => key === "__proto__" || !Object.prototype.propertyIsEnumerable.call(args, key),
  );
}

/**
 * The refusal the tool's own schema would have given, for a key it never gets to judge.
 *
 * The wording, the issue code and the wrapping are all borrowed rather than written: the
 * message is `unknownKeyMessage`, the same one the closed schema's error callback returns; the
 * JSON around it is `ZodError`'s own rendering of an `unrecognized_keys` issue; and the prefix
 * and error code are the SDK's `McpError`. What is left is one sentence of ours, and a test
 * that takes an ordinary unknown key's refusal, swaps the key name into it and demands this
 * one match character for character.
 */
function droppedKeyRefusal(
  id: unknown,
  name: string,
  shape: ZodRawShape,
  dropped: string[],
): JSONRPCMessage {
  const refusal = new McpError(
    ErrorCode.InvalidParams,
    `Input validation error: Invalid arguments for tool ${name}: ` +
      new ZodError([
        {
          code: "unrecognized_keys",
          keys: dropped,
          path: [],
          message: unknownKeyMessage(new Set(Object.keys(shape)), dropped),
        },
      ]).message,
  );
  return {
    jsonrpc: "2.0",
    id,
    result: { content: [{ type: "text", text: refusal.message }], isError: true },
  } as unknown as JSONRPCMessage;
}

/**
 * A response to send in this call's place, or undefined to let the message through.
 *
 * Which calls get looked at is the SDK's question and not one of ours: it takes `params` by
 * `typeof` and `arguments` by `isPlainObject`, so a class instance carrying a real `name` and an
 * argument list another realm parsed are both calls it runs and drops a key out of. Asked either
 * question our own way, those two went through unanswered.
 *
 * What is inspected is the inert copy of the argument list that the SDK will parse -- the same
 * copy, read from the caller once -- so "inspected" and "parsed" cannot be two different objects.
 * The decoding and the refusal are one read of the message (`readCall`); this is the other half
 * of its answer. Nothing here may throw, for the reason `readCall` gives.
 */
export function refuseDroppedKeys(
  message: JSONRPCMessage,
  shapes: Map<string, ZodRawShape>,
): JSONRPCMessage | undefined {
  return readCall(message, shapes).refusal;
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
 * It also answers one call itself, rather than passing it on: one carrying an argument the SDK
 * would drop on the way in (see keysTheSdkDrops). That refusal has to be written here for the
 * same reason the decoding does -- this is the last point at which the argument still exists.
 *
 * What it does not touch: the message stream in either direction beyond that one field and that
 * one refusal, the order of anything, or the inner transport's lifecycle. `start`, `send` and
 * `close` are the inner transport's, and `sessionId` is read from it live rather than copied,
 * because an HTTP transport assigns one during initialize.
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
        // One read of the message, and both answers come out of it: a call this refuses and a
        // call this decodes are the same read, of the same copy.
        const read = readCall(message, this.shapes);
        if (read.refusal !== undefined) {
          // Answered rather than forwarded: past this point the argument being refused is gone,
          // so the server it would be forwarded to could not refuse it.
          this.inner.send(read.refusal).catch((error: unknown) => this.onerror?.(error as Error));
          return;
        }
        this.onmessage?.(read.message, extra);
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
