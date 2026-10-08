import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { runInNewContext } from "node:vm";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { buildServer } from "../src/server";

/**
 * A call built out of live objects, and the one rule that makes it answerable: read once.
 *
 * `InMemoryTransport` hands the server the object the peer built, so a `params`, an argument list
 * or a value inside one need not have come from `JSON.parse`. It can be a class instance, an
 * object another realm parsed, an object carrying accessors, or a Proxy -- and a Proxy can answer
 * a question one way and then the other way, which is what makes reading a value twice a bug
 * rather than an inefficiency. Four rounds of review found four pairs of reads on this surface:
 * a key snapshot and the record's own enumeration; a dropped-key inspection and the SDK's parse;
 * a decode and the SDK's parse; a spread that dropped an inherited `name` the SDK would have read.
 *
 * So every field is read once, at the transport, into inert data, and everything after that --
 * the dropped-key refusal, the lax decoding, the SDK's own parse of `params` and of the tool's
 * arguments, and `jsonObject()` inside an object-valued parameter -- reads a copy this realm
 * built. These tests are that property: what ran is what was read, once, and the reads are
 * counted.
 */

/** Every request one test made. */
let sent: Array<{ url: string; method: string; body: string }> = [];

function mockGalaxy(): void {
  vi.spyOn(globalThis, "fetch").mockImplementation((async (input: unknown, init?: RequestInit) => {
    const url =
      typeof input === "string" ? input : input instanceof URL ? input.href : (input as Request).url;
    const method = input instanceof Request ? input.method : (init?.method ?? "GET");
    // openapi-fetch hands `fetch` a Request, so what was posted is on that rather than in init.
    const body =
      input instanceof Request
        ? await input.clone().text()
        : typeof init?.body === "string"
          ? init.body
          : "";
    sent.push({ url, method, body });
    const answer = url.includes("/api/version")
      ? { version_major: "26.1", version_minor: "26.1.1" }
      : url.includes("/api/users/current")
        ? { id: "u1", email: "u@example.org", username: "u" }
        : url.includes("/api/unprivileged_tools")
          ? { id: "ut1", uuid: "u-1" }
          : method === "POST"
            ? { id: "h9", name: "x" }
            : Array.from({ length: 10 }, (_, k) => ({ id: `h${k}`, name: `h${k}` }));
    return new Response(JSON.stringify(answer), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }) as unknown as typeof fetch);
}

beforeEach(() => {
  sent = [];
  mockGalaxy();
});

afterEach(() => vi.restoreAllMocks());

interface Reply {
  id?: unknown;
  result?: { isError?: boolean; content?: Array<{ text?: string }> };
  error?: { code: number; message: string };
}

/**
 * One message sent as the peer built it, envelope and all.
 *
 * No `Client`, because a client would build the message itself; the point of every test here is
 * the object the peer hands over -- which, over an `InMemoryTransport`, is handed to the server's
 * `onmessage` exactly as it was constructed. `wait` is how long to sit there for a reply: a
 * message the SDK cannot even recognise as a request gets none at all, and a row expecting that
 * says so instead of waiting out the runner's clock.
 */
async function rawMessage(message: unknown, wait = 5_000): Promise<Reply | undefined> {
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  const server = buildServer({ baseUrl: "https://g.example", apiKey: "K" });
  await server.connect(serverTransport);
  const replies: Reply[] = [];
  let arrived: () => void = () => {};
  const first = new Promise<void>((resolve) => (arrived = resolve));
  clientTransport.onmessage = (reply) => {
    replies.push(reply as Reply);
    arrived();
  };
  await clientTransport.start();
  await clientTransport.send(message as never);
  await Promise.race([first, new Promise((resolve) => setTimeout(resolve, wait))]);
  await server.close();
  return replies[0];
}

/** The same, with an ordinary envelope around the `params` a test cares about. */
const rawCall = async (params: unknown, wait = 5_000): Promise<Reply | undefined> =>
  rawMessage({ jsonrpc: "2.0", id: 1, method: "tools/call", params }, wait);

const rawText = (reply: Reply | undefined): string =>
  reply === undefined
    ? "no reply at all"
    : (reply.error?.message ?? (reply.result?.content ?? []).map((c) => c.text ?? "").join(""));

/** Whether the tool actually ran, read off what Galaxy was asked for rather than off the reply. */
const asked = (route: string): boolean => sent.some((r) => r.url.includes(route));

const REPRESENTATION =
  '{"class":"GalaxyUserTool","id":"t","version":"0.1","name":"t",' +
  '"shell_command":"echo hi","container":"python:3.12-slim"}';

/** The body `create_user_tool` posts for a representation, with whatever is in it. */
const posted = (route: string): string => {
  const hit = sent.find((r) => r.url.includes(route));
  expect(hit, `nothing was sent to ${route}; asked for ${JSON.stringify(sent.map((s) => s.url))}`).toBeDefined();
  return hit!.body;
};

describe("a value that answers differently the second time it is read", () => {
  /**
   * The representation whose key list gets shorter, and the field that used to fall out of it.
   *
   * A Proxy whose `ownKeys` deletes `note` the second time it is asked is a valid representation
   * by every question this surface asks: zod's `util.isPlainObject` takes it, and a record parsing
   * it once copies every key including `note`. What lost the field was asking twice -- the key
   * order was taken for the caller's own ordering, and the record then ran its own enumeration,
   * which was the second one. No `__proto__` anywhere in it; just a field the caller wrote and a
   * document Galaxy was posted without it.
   */
  it("posts every field of a representation that reports a shorter key list the second time", async () => {
    const text = `{${REPRESENTATION.slice(1, -1)},"note":"keep"}`;
    const target = JSON.parse(text) as Record<string, unknown>;
    let enumerations = 0;
    const representation = new Proxy(target, {
      ownKeys(t) {
        enumerations += 1;
        if (enumerations === 2) delete t.note;
        return Reflect.ownKeys(t);
      },
    });

    const reply = await rawCall({ name: "create_user_tool", arguments: { representation } });

    expect(reply?.result?.isError, rawText(reply)).toBe(false);
    expect(posted("/api/unprivileged_tools")).toBe(`{"src":"representation","representation":${text}}`);
    // One enumeration of the caller's object, and the record's own is of the copy.
    expect(enumerations).toBe(1);
  });

  /**
   * The representation that lies about its own TYPE rather than its contents.
   *
   * "Is this a record?" is zod's `util.isPlainObject`, which asks the value for its
   * `constructor` -- and it used to be asked of the caller's object, before the copy, because a
   * copy into a fresh `{}` would have said `Object` whatever it was copied from. That made two
   * reads with a copy in between, so a `constructor` answering `Object` and then `Date` was
   * checked as a record and stored as a Date: `8a6f67c` accepted this representation and posted
   * it, and the round that took the copy refused it with "expected record, received object". The
   * copy carries the prototype now, the question is the record's own and it is asked once, of
   * the copy -- so the answer is the one the caller gave, and it is given once.
   */
  it("posts a representation whose constructor says Object and would then say Date", async () => {
    let reads = 0;
    const representation = JSON.parse(REPRESENTATION) as Record<string, unknown>;
    Object.defineProperty(representation, "constructor", {
      get: () => (++reads === 1 ? Object : Date),
      enumerable: true,
      configurable: true,
    });

    const reply = await rawCall({ name: "create_user_tool", arguments: { representation } });

    expect(reply?.result?.isError, rawText(reply)).toBe(false);
    // `constructor` is an ordinary key to the record, copied like any other -- and a function is
    // not JSON, so what Galaxy is posted is the six fields the caller wrote.
    expect(posted("/api/unprivileged_tools")).toBe(`{"src":"representation","representation":${REPRESENTATION}}`);
    expect(reads).toBe(1);
  });

  /**
   * An `arguments` accessor with three answers, on a `params` the SDK takes and this surface
   * cannot rebuild by spreading.
   *
   * The SDK reads `params` by `typeof`, so a class instance carrying a real `name` is a call it
   * runs. Reading `arguments` off it twice -- once to ask whether it was an argument list, once to
   * count the keys the parser would drop -- left the SDK's own read as the third, and an accessor
   * that turned dirty on the third read got `__proto__` past a refusal that had already looked and
   * ran the tool without it. The value is read once now, so what was inspected is what runs: the
   * first answer, which is the one this asked for.
   */
  it("runs the argument list it read, once, when a later read would have said something else", async () => {
    let reads = 0;
    class Params {
      name = "create_history";
      get arguments(): Record<string, unknown> {
        reads += 1;
        return reads >= 3
          ? (JSON.parse('{"history_name":"x","__proto__":{}}') as Record<string, unknown>)
          : { history_name: "x" };
      }
    }

    const reply = await rawCall(new Params());

    expect(reply?.result?.isError, rawText(reply)).toBe(false);
    expect(reads).toBe(1);
    // The history that was created is the one the argument list this read asked for describes,
    // and the third answer was never requested by anybody.
    // bioblend posts a history's name as a form field, which is what the op does too.
    expect(sent.filter((r) => r.method === "POST").map((r) => r.body)).toEqual(["name=x"]);
  });

  /** And the same container answering with the key on the read that happens: refused, as promised. */
  it("refuses the dropped key when the one read is the read that carries it", async () => {
    class Params {
      name = "create_history";
      get arguments(): Record<string, unknown> {
        return JSON.parse('{"history_name":"x","__proto__":{}}') as Record<string, unknown>;
      }
    }

    const reply = await rawCall(new Params());

    expect(reply?.result?.isError).toBe(true);
    expect(rawText(reply)).toContain('Unrecognized parameter: \\"__proto__\\"');
    expect(sent.filter((r) => r.method === "POST")).toEqual([]);
  });

  /**
   * A `params` that keeps its `name` on a prototype, which the SDK reads and a spread lost.
   *
   * The SDK reads `params.name` by property access, so this is a call it runs -- and it ran on
   * `8a6f67c`, where a `params` like this failed the old container test and went through
   * untouched. Rebuilding `params` by spreading it dropped the inherited field and answered
   * "expected string, received undefined" to a call that used to work. So `name` and `arguments`
   * are read the way the SDK reads them, once each, and carried across by hand.
   */
  it("runs a call whose params keeps its name on a prototype", async () => {
    const params = Object.assign(Object.create({ name: "get_user" }) as object, { arguments: {} });

    const reply = await rawCall(params);

    expect(reply?.result?.isError, rawText(reply)).toBe(false);
    expect(rawText(reply)).toContain("Retrieved user info for 'u'");
    expect(asked("/api/users/current")).toBe(true);
  });

  /**
   * The fallback that forwarded the caller's object, and the third answer it let through.
   *
   * `arguments` reads as `undefined`, which is a field the peer wrote and did not fill, so the
   * call goes on without an argument list rather than with an empty one. What used to go on was
   * the MESSAGE THAT ARRIVED -- so the SDK asked the accessor again, twice, and the answer it got
   * carried a `__proto__` that its record dropped, running the tool on an argument list nobody
   * here had ever seen. The copy goes on now, holding the one answer this read.
   */
  it("hands on the arguments it read, not the object that would answer again", async () => {
    let reads = 0;
    class Params {
      name = "list_history_ids";
      get arguments(): Record<string, unknown> | undefined {
        reads += 1;
        return reads === 1 ? undefined : (JSON.parse('{"__proto__":{}}') as Record<string, unknown>);
      }
    }

    const reply = await rawCall(new Params());

    expect(reads).toBe(1);
    expect(reply?.result?.isError === true || reply?.error !== undefined, rawText(reply)).toBe(true);
    expect(asked("/api/histories")).toBe(false);
  });

  /**
   * And the same for the id, which the refusal needs and the rebuild used to read again.
   *
   * A message whose `id` answers `undefined` and then `1` had its dropped key waved through: the
   * refusal looked at the first answer, decided there was nobody to reply to, and the rebuild
   * then read the second answer and put a real id on the message the SDK ran. One read now, and
   * the id the refusal was told about is the id the copy carries -- so a message that says it has
   * none is a message nothing answers and nothing runs.
   */
  it("reads the id once, so a second answer cannot turn a notification into a call", async () => {
    let reads = 0;
    const message = {
      jsonrpc: "2.0",
      method: "tools/call",
      params: { name: "list_history_ids", arguments: JSON.parse('{"__proto__":{}}') as unknown },
    };
    Object.defineProperty(message, "id", { get: () => (++reads === 1 ? undefined : 1), enumerable: true });

    const reply = await rawMessage(message, 500);

    expect(reads).toBe(1);
    expect(reply, rawText(reply)).toBeUndefined();
    expect(asked("/api/histories")).toBe(false);
  });
});

/**
 * The eight fields the SDK's own schemas declare, read the way those schemas read them.
 *
 * `JSONRPCRequestSchema` declares `jsonrpc`, `id`, `method` and `params`;
 * `CallToolRequestParamsSchema` declares `name`, `arguments`, `_meta` and `task`. A zod object
 * reads a field it declares by PROPERTY ACCESS, so a non-enumerable own one and an inherited one
 * are both fields it sees -- and a rebuild that carried only the own enumerable keys was handing
 * the SDK a different message from the one that arrived. Each of these ran on `8a6f67c`, which
 * left such a message alone for the SDK to read.
 */
describe("a field the SDK declares, however the caller attached it", () => {
  it("runs a call whose id is a non-enumerable own property", async () => {
    class Params {
      name = "get_user";
      arguments = {};
    }
    const message = { jsonrpc: "2.0", method: "tools/call", params: new Params() };
    Object.defineProperty(message, "id", { value: 1, enumerable: false });

    const reply = await rawMessage(message);

    expect(reply?.result?.isError, rawText(reply)).toBe(false);
    expect(reply?.id).toBe(1);
    expect(asked("/api/users/current")).toBe(true);
  });

  /**
   * A `task` the SDK would refuse, hidden where a spread could not see it.
   *
   * `CallToolRequestParamsSchema` takes `task` as `{ttl?: number}`, so `{ttl: "bad"}` is a call
   * it turns down -- and it turned this one down on `8a6f67c`, which never touched a `params`
   * like this. Carrying only the own enumerable keys quietly dropped the field and ran the tool
   * instead, which is the one direction this must never go: a call main refused, running.
   */
  it("refuses a task the caller hid on the params, as it was refused before", async () => {
    class Params {
      name = "get_user";
      arguments = {};
    }
    const params = new Params();
    Object.defineProperty(params, "task", { value: { ttl: "bad" }, enumerable: false });

    const reply = await rawCall(params);

    expect(reply?.result?.isError === true || reply?.error !== undefined, rawText(reply)).toBe(true);
    expect(asked("/api/users/current")).toBe(false);
  });

  it("refuses a task the caller left on the prototype, for the same reason", async () => {
    const params = Object.assign(Object.create({ task: { ttl: "bad" } }) as object, {
      name: "get_user",
      arguments: {},
    });

    const reply = await rawCall(params);

    expect(reply?.result?.isError === true || reply?.error !== undefined, rawText(reply)).toBe(true);
    expect(asked("/api/users/current")).toBe(false);
  });

  /**
   * And `_meta`, the fourth field, which the envelope itself declares.
   *
   * `JSONRPCRequestSchema` takes `params` as a loose `{_meta}`, so a progress token that is not a
   * string or a number makes the whole message something the SDK does not recognise as a request
   * -- it answers nothing and runs nothing. That is what a caller writing this as JSON TEXT has
   * always got, and it is what an in-process caller gets now however the field was attached:
   * carrying only the own enumerable keys dropped an inherited one and ran the tool, which is the
   * two kinds of caller being told different things about the same message.
   */
  it("carries a _meta the caller left on the prototype", async () => {
    const meta = { progressToken: {} };
    const written = { name: "get_user", arguments: {}, _meta: meta };
    const inherited = Object.assign(Object.create({ _meta: meta }) as object, {
      name: "get_user",
      arguments: {},
    });

    for (const params of [written, inherited]) {
      sent = [];
      const reply = await rawCall(params, 500);
      expect(reply, rawText(reply)).toBeUndefined();
      expect(asked("/api/users/current")).toBe(false);
    }
  });
});

/**
 * Every trap invocation, counted, for one successful call.
 *
 * This is the property the five rounds were circling: not "the second read agrees with the first"
 * but "there is no second read". `params`, the argument list and the representation inside it are
 * each a Proxy that counts what is asked of it, and the expectation is written out in full -- one
 * prototype, one `ownKeys`, one descriptor and one value read per key, and one property access
 * per field the SDK's schemas declare. `constructor` is not on any of these lists, because the
 * plainness question is asked of the copy: the prototype came across with it, so the copy has
 * the same answer and the caller's object is never asked.
 */
describe("the reads a successful call makes of the objects the caller handed over", () => {
  function counting<T extends object>(target: T): { proxy: T; reads: string[] } {
    const reads: string[] = [];
    const proxy = new Proxy(target, {
      ownKeys(t) {
        reads.push("ownKeys");
        return Reflect.ownKeys(t);
      },
      getOwnPropertyDescriptor(t, key) {
        reads.push(`descriptor:${String(key)}`);
        return Reflect.getOwnPropertyDescriptor(t, key);
      },
      get(t, key, receiver) {
        reads.push(`get:${String(key)}`);
        return Reflect.get(t, key, receiver) as unknown;
      },
      has(t, key) {
        reads.push(`has:${String(key)}`);
        return Reflect.has(t, key);
      },
      getPrototypeOf(t) {
        reads.push("getPrototypeOf");
        return Reflect.getPrototypeOf(t);
      },
    });
    return { proxy, reads };
  }

  it("reads each of them exactly once, and nothing twice", async () => {
    const representation = counting(JSON.parse(REPRESENTATION) as Record<string, unknown>);
    const args = counting({ representation: representation.proxy } as Record<string, unknown>);
    const params = counting({ name: "create_user_tool", arguments: args.proxy });

    const reply = await rawCall(params.proxy);
    expect(reply?.result?.isError, rawText(reply)).toBe(false);
    expect(posted("/api/unprivileged_tools")).toBe(`{"src":"representation","representation":${REPRESENTATION}}`);

    // `params`: the four fields the SDK's own schema declares are read here by property access,
    // once each, so the copy handed on carries the values this read. `_meta` and `task` are not
    // here, and a field that reads as undefined is asked `in` once to tell absent from written
    // as undefined -- which runs no getter. Then one prototype and one enumeration for the
    // extras, of which there are none, so no descriptor is asked for at all.
    expect(params.reads).toEqual([
      "get:name",
      "get:arguments",
      "get:_meta",
      "has:_meta",
      "get:task",
      "has:task",
      "getPrototypeOf",
      "ownKeys",
    ]);

    // The argument list: its prototype, one enumeration, one descriptor and one value per key.
    // The decoding, the dropped-key count, the plainness question and the SDK's own record parse
    // all read the copy that came out of this.
    expect(args.reads).toEqual([
      "getPrototypeOf",
      "ownKeys",
      "descriptor:representation",
      "get:representation",
    ]);

    // And the value inside the parameter, at its own level, when `jsonObject()` reads it: the same
    // questions, once per key, and no fourth from the record that parses the copy.
    expect(representation.reads).toEqual([
      "getPrototypeOf",
      "ownKeys",
      ...["class", "id", "version", "name", "shell_command", "container"].flatMap((key) => [
        `descriptor:${key}`,
        `get:${key}`,
      ]),
    ]);

    // Said once more as the property rather than as a list, because this is the one that used to
    // be asked twice: at most one `constructor` read per container, and in fact none, since the
    // question that asks for it is now asked of the copy.
    for (const reads of [params.reads, args.reads, representation.reads]) {
      expect(reads.filter((read) => read === "get:constructor")).toEqual([]);
    }
  });

  /**
   * And nothing reads them again after the copy is taken -- asserted by making a second read
   * impossible rather than by counting it.
   *
   * Each container answers normally until its last expected read and then throws on any trap at
   * all. A call that still succeeds is a call where every later question went to the copy: the
   * dropped-key count, the lax decoding, zod's plainness gate, the SDK's parse of `params` and of
   * the arguments, and the record inside `jsonObject()`.
   */
  it("succeeds against containers that refuse to be read a second time", async () => {
    const fuse = <T extends object>(target: T, last: string): T => {
      let spent = false;
      const guard = (op: string): void => {
        if (spent) throw new Error(`${op}: read again after the copy was taken`);
        if (op === last) spent = true;
      };
      return new Proxy(target, {
        ownKeys(t) {
          guard("ownKeys");
          return Reflect.ownKeys(t);
        },
        getOwnPropertyDescriptor(t, key) {
          guard(`descriptor:${String(key)}`);
          return Reflect.getOwnPropertyDescriptor(t, key);
        },
        get(t, key, receiver) {
          guard(`get:${String(key)}`);
          return Reflect.get(t, key, receiver) as unknown;
        },
        has(t, key) {
          guard(`has:${String(key)}`);
          return Reflect.has(t, key);
        },
        getPrototypeOf(t) {
          guard("getPrototypeOf");
          return Reflect.getPrototypeOf(t);
        },
      });
    };

    const representation = fuse(JSON.parse(REPRESENTATION) as Record<string, unknown>, "get:container");
    const args = fuse({ representation } as Record<string, unknown>, "get:representation");
    const params = fuse({ name: "create_user_tool", arguments: args }, "ownKeys");

    const reply = await rawCall(params);

    expect(reply?.result?.isError, rawText(reply)).toBe(false);
    expect(posted("/api/unprivileged_tools")).toBe(`{"src":"representation","representation":${REPRESENTATION}}`);
  });
});

/**
 * The containers an in-process peer can hand over as `params`, against what `8a6f67c` did with
 * each one -- measured, by running this table with that commit's `laxenCall` and its
 * prototype-identity container test dropped into this tree.
 *
 * Every row answers as it did there but one, and that one is an exception the rule owns: a
 * `params` whose `name` is a non-enumerable OWN property ran nowhere before, because the SDK read
 * a field the rebuild's spread had dropped and said "expected string, received undefined". The
 * name is read the way the SDK reads it now, so the call runs. The row above it is the opposite
 * story and not an exception at all: a `name` on a prototype ran on `8a6f67c`, the rebuild in
 * between broke it, and it runs again.
 */
describe("a params container, whatever it is made of", () => {
  const getUser = { name: "get_user", arguments: {} };

  const hidden = (): object => {
    const params: Record<string, unknown> = { arguments: {} };
    Object.defineProperty(params, "name", { value: "get_user", enumerable: false });
    return params;
  };

  const withAccessor = (): object => {
    const params: Record<string, unknown> = { name: "get_user" };
    Object.defineProperty(params, "arguments", { get: () => ({}), enumerable: true });
    return params;
  };

  it.each([
    ["a plain object", () => ({ ...getUser }), "runs"],
    ["a null-prototype object", () => Object.assign(Object.create(null) as object, getUser), "runs"],
    [
      "an object another realm parsed",
      () => runInNewContext("JSON.parse(text)", { text: JSON.stringify(getUser) }) as object,
      "runs",
    ],
    [
      "a class instance",
      () =>
        new (class Params {
          name = "get_user";
          arguments = {};
        })(),
      "runs",
    ],
    // Exception 1: `8a6f67c` ran this, the rebuild in between refused it, it runs again.
    [
      "its name on a prototype",
      () => Object.assign(Object.create({ name: "get_user" }) as object, { arguments: {} }),
      "runs",
    ],
    // Exception 2: `8a6f67c` ran this because the SDK read a `name` the spread had dropped.
    ["a non-enumerable own name", hidden, "runs"],
    ["a Proxy of a plain object", () => new Proxy({ ...getUser }, {}), "runs"],
    ["an accessor for arguments", withAccessor, "runs"],
    ["a Date", () => new Date(0), "refused"],
    ["a Map", () => new Map(Object.entries(getUser)), "refused"],
    // Neither of the last two is a `params` the SDK's envelope schema will take at all, so the
    // message is not a request and nobody answers it. Same on `8a6f67c`: this surface never
    // touched either one.
    ["an array", () => Object.assign([], getUser), "silent"],
    ["a string", () => "not params at all", "silent"],
  ] as Array<[string, () => unknown, "runs" | "refused" | "silent"]>)(
    "%s: the call %s",
    async (_what, make, outcome) => {
      const reply = await rawCall(make(), outcome === "silent" ? 500 : 5_000);
      expect(asked("/api/users/current"), rawText(reply)).toBe(outcome === "runs");
      if (outcome === "runs") expect(reply?.result?.isError, rawText(reply)).toBe(false);
      else if (outcome === "refused")
        expect(reply?.result?.isError === true || reply?.error !== undefined, rawText(reply)).toBe(true);
      else expect(reply, rawText(reply)).toBeUndefined();
    },
  );
});

/**
 * The same for the argument list, where the container question is the record parser's.
 *
 * `list_history_ids` takes `limit`, and `"5"` is a value only the lax decoding accepts -- so a
 * row that pages five histories is a row where the container was decoded, and a row refused with
 * "expected number" is one where it was not. Measured against `8a6f67c` the same way, one row
 * differs: an argument list another realm parsed is decoded now rather than refused, which is the
 * container predicate becoming zod's own and is in the changelog as a caller-visible change.
 *
 * The last row is the asymmetry with `params` written down: the SDK reads `params.name` by
 * property access and an argument list by `Reflect.ownKeys`, so a `limit` on a prototype is not an
 * argument at all, to either surface, before or after.
 */
describe("an arguments container, whatever it is made of", () => {
  const withAccessor = (): Record<string, unknown> => {
    const args: Record<string, unknown> = {};
    Object.defineProperty(args, "limit", { get: () => "5", enumerable: true, configurable: true });
    return args;
  };

  it.each([
    ["a plain object", () => ({ limit: "5" }), 5],
    ["a null-prototype object", () => Object.assign(Object.create(null) as object, { limit: "5" }), 5],
    [
      "an object another realm parsed",
      () => runInNewContext("JSON.parse(text)", { text: '{"limit":"5"}' }) as Record<string, unknown>,
      5,
    ],
    ["a Proxy of a plain object", () => new Proxy({ limit: "5" }, {}), 5],
    ["an accessor for limit", withAccessor, 5],
    // Not a record to zod, so not decoded and not inspected: the SDK refuses the container.
    ["a class instance", () => new (class Args { limit = "5" })(), undefined],
    ["a Date", () => new Date(0), undefined],
    ["a Map", () => new Map([["limit", "5"]]), undefined],
    ["an array", () => [5], undefined],
    ["a string", () => "5", undefined],
    // A record reads own keys only, so this is a call with no arguments -- which `list_history_ids`
    // is allowed to be, and it pages on its default.
    ["its limit on a prototype", () => Object.create({ limit: "5" }) as Record<string, unknown>, 10],
  ] as Array<[string, () => unknown, number | undefined]>)(
    "%s: pages %s histories",
    async (_what, make, items) => {
      const reply = await rawCall({ name: "list_history_ids", arguments: make() });
      if (items === undefined) {
        expect(reply?.result?.isError === true || reply?.error !== undefined, rawText(reply)).toBe(true);
        expect(asked("/api/histories")).toBe(false);
        return;
      }
      expect(reply?.result?.isError, rawText(reply)).toBe(false);
      const page = JSON.parse(rawText(reply)) as { data: unknown[] };
      expect(page.data).toHaveLength(items);
    },
  );
});

/**
 * The edges of that rule, pinned as edges.
 *
 * Read once is a promise about the eight fields the SDK's schemas declare and the own enumerable
 * keys of the containers they arrive in. It is not a promise about everything an in-process peer
 * can build, and this is the three places it stops -- here so that the contract written down in
 * `materializeOnce` and in the changelog says what the code does, and so that widening any of
 * them is a test failure rather than a discovery.
 *
 * A key JSON text cannot write gets no promise either way: the dropped-key count reads
 * `getOwnPropertyNames`, which does not see a symbol, so a hidden one goes the way the SDK's
 * record takes it and the call runs. A value INSIDE `_meta` is handed to the SDK as it came,
 * because nothing here parses it, so a getter in there answers as often as the SDK asks -- how
 * often is the SDK's business, and the assertion says only that this is not read-once ground.
 * And the masquerade stays refused: the record question is asked of the copy, the copy is built
 * on the prototype the value really had, and a class instance is not a record however it answers
 * when it is asked for its `constructor`.
 */
describe("what the boundary does not promise", () => {
  it("drops a hidden symbol key, rereads inside `_meta`, and refuses a masquerading record", async () => {
    const args: Record<string, unknown> = {};
    Object.defineProperty(args, Symbol("hidden"), { value: 1 });

    const hidden = await rawCall({ name: "get_user", arguments: args });

    expect(hidden?.result?.isError, rawText(hidden)).toBe(false);
    expect(asked("/api/users/current")).toBe(true);

    sent = [];
    let metaReads = 0;
    const meta = {
      get progressToken(): string {
        metaReads += 1;
        return "p1";
      },
    };

    const withMeta = await rawCall({ name: "get_user", arguments: {}, _meta: meta });

    expect(withMeta?.result?.isError, rawText(withMeta)).toBe(false);
    expect(metaReads).toBeGreaterThanOrEqual(1);

    sent = [];
    class Rep {
      constructor(fields: Record<string, unknown>) {
        Object.assign(this, fields);
      }
    }
    const masked = new Proxy(new Rep(JSON.parse(REPRESENTATION) as Record<string, unknown>), {
      get: (target, key, receiver) =>
        key === "constructor" ? Object : (Reflect.get(target, key, receiver) as unknown),
    });

    const masquerade = await rawCall({ name: "create_user_tool", arguments: { representation: masked } });

    expect(rawText(masquerade)).toContain("expected record, received Rep");
    expect(asked("/api/unprivileged_tools")).toBe(false);
  });
});
