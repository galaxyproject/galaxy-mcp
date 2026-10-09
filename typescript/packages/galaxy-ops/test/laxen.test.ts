import { describe, it, expect } from "vitest";
import { runInNewContext } from "node:vm";
import { z } from "zod";
import { isPlainObject, laxenLikePydantic, materializeOnce } from "../src/laxen";

/**
 * The decoding pydantic does for a tool call's arguments, unit by unit.
 *
 * Every row here was measured against the installed pydantic with
 * `TypeAdapter(int | None)` and `TypeAdapter(bool)`, in both `validate_python` and
 * `validate_json` -- they agree -- and the same rows are driven through both surfaces in
 * `galaxy-mcp/test/lax-arguments.test.ts` and `galaxy-cli/test/lax-arguments.test.ts`.
 */
const shape = {
  limit: z.number().int().optional(),
  version: z.number().int().nullish(),
  flag: z.boolean().optional(),
  ratio: z.number().optional(),
  label: z.string().optional(),
};

const decodes = (key: string, given: unknown): unknown => laxenLikePydantic(shape, { [key]: given })[key];

describe("an integer argument", () => {
  it.each([
    ["5", 5],
    [" 5 ", 5],
    ["+5", 5],
    ["-5", -5],
    ["05", 5],
    ["5.0", 5],
    ["5.00", 5],
    ["1_000", 1000],
    [5.0, 5],
    [-0.0, 0],
    [1e2, 100],
    [true, 1],
    [false, 0],
  ])("reads %j as %i", (given, want) => {
    expect(decodes("limit", given)).toBe(want);
    expect(decodes("version", given)).toBe(want);
  });

  it.each([["5.5"], [5.5], ["1e2"], ["0x10"], [""], ["abc"], ["٥"], ["1__0"], ["_1"], ["1_"], ["5."], ["."], [[1]], [{ a: 1 }], [Number.NaN], [Number.POSITIVE_INFINITY]])(
    "leaves %j for the schema to refuse",
    (given) => {
      expect(decodes("limit", given)).toBe(given);
    },
  );

  /**
   * Two quirks of pydantic's own parser that this does not copy, and neither did anything
   * before it: a leading zero makes it forgiving about what follows, so "0__5" is 5 and
   * "0-5" is -5 over there and a refusal here. Worth knowing about, not worth reproducing --
   * nothing sends them on purpose.
   */
  it.each([["0__5"], ["0-5"]])("refuses pydantic's leading-zero quirk %j", (given) => {
    expect(decodes("limit", given)).toBe(given);
  });

  it("agrees with pydantic on the ordinary leading-zero forms", () => {
    // "0_5" and "00_5" are 5 on both sides; it is the doubled underscore and the embedded
    // sign that only pydantic's parser lets through.
    expect(decodes("limit", "0_5")).toBe(5);
    expect(decodes("limit", "00_5")).toBe(5);
  });

  it("leaves an integer JavaScript cannot hold, rather than changing it", () => {
    // pydantic keeps 9007199254740993 exactly; Number() would hand back ...92.
    expect(decodes("limit", "9007199254740993")).toBe("9007199254740993");
  });

  it("leaves null and undefined for the schema to rule on", () => {
    expect(decodes("limit", null)).toBeNull();
    expect(laxenLikePydantic(shape, {})).toEqual({});
  });

  it("does not touch a field that is not an integer", () => {
    expect(decodes("ratio", "5")).toBe("5");
    expect(decodes("label", "5")).toBe("5");
  });
});

/**
 * The whitespace pydantic strips, code point by code point, because it is not the whitespace
 * `String.prototype.trim` strips. Measured with `TypeAdapter(int | None)`: everything in the
 * first list is trimmed there and everything in the second is refused.
 *
 * The two that disagree with `trim()` are the reason this is a table and not a call to it:
 * U+FEFF is stripped by JavaScript and refused by pydantic, and U+0085 is the other way
 * round.
 */
describe("the whitespace around an integer", () => {
  it.each([
    ["\t", "tab"],
    ["\n", "line feed"],
    ["\v", "vertical tab"],
    ["\f", "form feed"],
    ["\r", "carriage return"],
    [" ", "space"],
    ["", "next line -- trim() does NOT strip this one"],
    [" ", "no-break space"],
    [" ", "ogham space mark"],
    [" ", "en quad"],
    [" ", "figure space"],
    [" ", "hair space"],
    [" ", "line separator"],
    [" ", "paragraph separator"],
    [" ", "narrow no-break space"],
    [" ", "medium mathematical space"],
    ["　", "ideographic space"],
  ])("strips %j (%s)", (space) => {
    expect(decodes("limit", `${space}5${space}`)).toBe(5);
  });

  it.each([
    ["﻿", "zero width no-break space -- trim() strips this one, pydantic refuses it"],
    ["᠎", "mongolian vowel separator"],
    ["​", "zero width space"],
  ])("does not strip %j (%s)", (space) => {
    expect(decodes("limit", `${space}5${space}`)).toBe(`${space}5${space}`);
  });
});

describe("a boolean argument", () => {
  it.each([
    ["true", true],
    ["false", false],
    ["True", true],
    ["FALSE", false],
    ["t", true],
    ["f", false],
    ["yes", true],
    ["no", false],
    ["y", true],
    ["n", false],
    ["on", true],
    ["off", false],
    ["Off", false],
    ["1", true],
    ["0", false],
    [1, true],
    [0, false],
    [1.0, true],
    [0.0, false],
  ])("reads %j as %j", (given, want) => {
    expect(decodes("flag", given)).toBe(want);
  });

  it.each([[2], [-1], [""], [" true "], ["maybe"], [[true]], [{}]])(
    "leaves %j for the schema to refuse",
    (given) => {
      expect(decodes("flag", given)).toBe(given);
    },
  );
});

/**
 * One definition of "plain object", used by the decoder and by the surface that feeds it, and it
 * is zod's own: the question is whether the SDK's parser will go on to take the value as a
 * record, so asking it any other way means deciding something about a container zod has already
 * decided about.
 *
 * `typeof` is not it: everything in the second list answers "object" and none of them is a set of
 * arguments, and a transport that carries references rather than JSON text can deliver any of
 * them. The prototype's identity is not it either -- an object another realm parsed has that
 * realm's `Object.prototype`, and it is an argument list like any other.
 */
describe("what counts as a plain object", () => {
  it.each([
    [{}, "an object literal"],
    [{ limit: 5 }, "one with a key in it"],
    [Object.create(null) as object, "a null prototype"],
    // Taken by zod's record parser and so taken here: an in-process caller on an
    // `InMemoryTransport` really can hand over an object it parsed somewhere else.
    [runInNewContext("JSON.parse('{\"limit\":\"5\"}')") as object, "a cross-realm object"],
    [runInNewContext("Object.create(null)") as object, "a cross-realm null prototype"],
    // A prototype that is itself a plain object: the record copies its own enumerable keys and
    // ignores the rest, which is what spreading it does too.
    [Object.create({ inherited: 1 }) as object, "a plain object as the prototype"],
  ] as Array<[unknown, string]>)("takes %s (%s)", (value) => {
    expect(isPlainObject(value)).toBe(true);
  });

  it.each([
    [[], "an array"],
    [new Date(0), "a Date"],
    [new Map(), "a Map"],
    [new Set(), "a Set"],
    // eslint-disable-next-line no-new-wrappers
    [new Number(5), "a boxed number"],
    // eslint-disable-next-line no-new-wrappers
    [new String("x"), "a boxed string"],
    [new (class Args {})(), "a class instance"],
    [runInNewContext("new Date(0)") as object, "a cross-realm Date"],
    [runInNewContext("[1]") as object, "a cross-realm array"],
    [null, "null"],
    [5, "a number"],
    ["x", "a string"],
  ] as Array<[unknown, string]>)("refuses %s (%s)", (value) => {
    expect(isPlainObject(value)).toBe(false);
  });

  it.each([
    [new Date(0), "a Date"],
    [new Map([["limit", 5]]), "a Map"],
    [new (class Args { limit = 5 })(), "a class instance"],
  ] as Array<[unknown, string]>)(
    "hands %s back rather than spreading it",
    (notArguments) => {
      expect(laxenLikePydantic(shape, notArguments as Record<string, unknown>)).toBe(notArguments);
    },
  );

  it("decodes what another realm parsed, the same as one parsed here", () => {
    const elsewhere = runInNewContext("JSON.parse('{\"limit\":\"5\"}')") as Record<string, unknown>;
    // The probe is only a probe if the object really is foreign.
    expect(Object.getPrototypeOf(elsewhere)).not.toBe(Object.prototype);
    expect(laxenLikePydantic(shape, elsewhere)).toEqual({ limit: 5 });
    expect(laxenLikePydantic(shape, { limit: "5" })).toEqual({ limit: 5 });
  });
});

/**
 * The boundary read itself, at the unit level: what it copies, and how many questions it asks.
 *
 * Everything on these surfaces that takes an object from a caller goes through here, so the two
 * things worth pinning are that the copy answers the way the original did to every question the
 * parsers downstream ask of it, and that the original was asked once.
 */
describe("a caller's object, read once", () => {
  /** Every trap invocation, in order. */
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
    });
    return { proxy, reads };
  }

  it("asks one enumeration, one descriptor and one value per key", () => {
    const { proxy, reads } = counting({ a: 1, b: 2 });
    expect(materializeOnce(proxy)).toEqual({ a: 1, b: 2 });
    expect(reads).toEqual(["ownKeys", "descriptor:a", "get:a", "descriptor:b", "get:b"]);
  });

  it("runs an enumerable getter once and stores what it returned", () => {
    let reads = 0;
    const value = {};
    Object.defineProperty(value, "a", {
      get: () => ++reads,
      enumerable: true,
      configurable: true,
    });
    const copy = materializeOnce(value) as Record<string, unknown>;
    expect(reads).toBe(1);
    // Data, not an accessor: reading it again is reading the copy, and it says the same thing.
    expect([copy.a, copy.a]).toEqual([1, 1]);
    expect(Object.getOwnPropertyDescriptor(copy, "a")).toEqual({
      value: 1,
      writable: true,
      enumerable: true,
      configurable: true,
    });
  });

  it("copies __proto__ as a property rather than setting one", () => {
    const copy = materializeOnce(JSON.parse('{"__proto__":{"direct":1},"keep":2}')) as Record<string, unknown>;
    expect(Object.getPrototypeOf(copy)).toBe(Object.prototype);
    expect(Object.hasOwn(copy, "__proto__")).toBe(true);
    expect(Reflect.ownKeys(copy)).toEqual(["__proto__", "keep"]);
  });

  it("keeps a symbol key, for the record to refuse", () => {
    const key = Symbol("unexpected");
    const copy = materializeOnce({ a: 1, [key]: 2 }) as Record<string | symbol, unknown>;
    expect(Reflect.ownKeys(copy)).toEqual(["a", key]);
    expect(copy[key]).toBe(2);
  });

  /**
   * A key the parsers skip while copying is carried by name and left empty on purpose: it is a key
   * that arrived, and the refusal in `galaxy-mcp`'s transport is the last place it can be seen, so
   * the copy has to show it. Reading what it held would run a getter neither parser runs.
   */
  it("carries a non-enumerable key by name, with no value and no read", () => {
    let reads = 0;
    const value: Record<string, unknown> = { a: 1 };
    Object.defineProperty(value, "hush", { get: () => ++reads, enumerable: false, configurable: true });
    const copy = materializeOnce(value) as Record<string, unknown>;
    expect(reads).toBe(0);
    expect(Object.getOwnPropertyNames(copy)).toEqual(["a", "hush"]);
    expect(Object.prototype.propertyIsEnumerable.call(copy, "hush")).toBe(false);
    expect(copy.hush).toBeUndefined();
    // Which is what the copy is for: enumerating it the way zod does finds what the original
    // would have offered, and nothing else.
    expect(Object.keys(copy)).toEqual(["a"]);
  });

  it("leaves out a key its caller reads for itself, rather than reading it twice", () => {
    const { proxy, reads } = counting({ name: "get_user", arguments: {} });
    expect(materializeOnce(proxy, ["name", "arguments"])).toEqual({});
    expect(reads).toEqual(["ownKeys"]);
  });

  it("hands back anything that is not an object, because there is nothing to read", () => {
    const fn = () => 1;
    for (const value of ["x", 5, true, null, undefined]) {
      expect(materializeOnce(value)).toBe(value);
    }
    // A function is not an object to `typeof`, and nothing here hands one to a parser.
    expect(materializeOnce(fn)).toBe(fn);
  });

  /**
   * An array goes back as it came, for the same reason a primitive does: no parser downstream
   * reads the contents of one. Both refuse it by type and name the type, and a copy carrying
   * `Array.prototype` is not an array to `Array.isArray` -- so the only thing a copy could change
   * is the word in a refusal that was going to happen anyway.
   */
  it("hands back an array as it came", () => {
    const value = [1, 2];
    expect(materializeOnce(value)).toBe(value);
  });

  /**
   * The copy keeps the prototype, because "is this a record?" is a question about the prototype
   * and it has to have the same answer before and after. Copying into a fresh `{}` made that
   * answer `Object` for everything, so the question had to be asked of the caller's object
   * first -- a second read, and a `constructor` getter answering differently sat between the two.
   */
  it("keeps the prototype the value arrived with, so the copy is the same kind of thing", () => {
    for (const value of [new Date(0), new Map(), new (class Args {})()]) {
      const copy = materializeOnce(value);
      expect(Object.getPrototypeOf(copy)).toBe(Object.getPrototypeOf(value));
      expect(isPlainObject(copy)).toBe(false);
    }

    // And the two that are records stay records: another realm's plain object, and one with no
    // prototype at all.
    const crossRealm = runInNewContext("JSON.parse(text)", { text: '{"a":1}' }) as object;
    expect(isPlainObject(materializeOnce(crossRealm))).toBe(true);
    expect(Object.getPrototypeOf(materializeOnce(Object.create(null) as object))).toBeNull();
  });

  /**
   * Which is the whole of it: a `constructor` the caller wrote is an ordinary key, read once and
   * stored as data, and the plainness question asked of the copy gets that one answer.
   */
  it("reads an own constructor once, like any other key", () => {
    let reads = 0;
    const value = {};
    Object.defineProperty(value, "constructor", {
      get: () => (++reads === 1 ? Object : Date),
      enumerable: true,
      configurable: true,
    });
    const copy = materializeOnce(value);
    expect(reads).toBe(1);
    expect(isPlainObject(copy)).toBe(true);
    expect(isPlainObject(copy)).toBe(true);
    expect(reads).toBe(1);
  });

  /**
   * Only the top level. A nested value is the caller's own data on its way somewhere, and whoever
   * parses it reads it once at its own level -- which is what `jsonObject()` does inside an
   * object-valued parameter.
   */
  it("does not go inside a value", () => {
    const nested = { deep: 1 };
    const copy = materializeOnce({ nested }) as Record<string, unknown>;
    expect(copy.nested).toBe(nested);
  });

  it("copies what a Proxy said the first time, however it answers after that", () => {
    const target: Record<string, unknown> = { a: 1, b: 2 };
    let enumerations = 0;
    const proxy = new Proxy(target, {
      ownKeys(t) {
        enumerations += 1;
        if (enumerations === 2) delete t.b;
        return Reflect.ownKeys(t);
      },
    });
    const copy = materializeOnce(proxy) as Record<string, unknown>;
    // The second enumeration never happens here, so `b` is in the copy -- and a parser that
    // enumerates the copy finds it there too, which is the whole point.
    expect(enumerations).toBe(1);
    expect(Object.keys(copy)).toEqual(["a", "b"]);
  });
});
