import { describe, it, expect } from "vitest";
import { runInNewContext } from "node:vm";
import { z } from "zod";
import { isJsonObjectSchema, jsonObject } from "../src/json-object";

/**
 * The parameter type that keeps a key called `__proto__` and changes nothing else.
 *
 * `z.record(z.string(), z.unknown())` is what all five of these parameters were on main, spelled
 * exactly as it is spelled here, and it parses by copying the object into a fresh one -- skipping
 * `__proto__` while it copies. So the record below is not a stand-in for the old behaviour, it IS
 * the old behaviour, and nearly every test in this file is a comparison against it: same decision,
 * same issues, same copy, one key more.
 */
describe("jsonObject", () => {
  const record = z.record(z.string(), z.unknown());

  /**
   * The caller's own `__proto__`, when they wrote one the record would have copied but for its
   * name: own, enumerable and holding a value are what the record asks of every key it copies.
   */
  function ownProto(input: unknown): { value: unknown } | undefined {
    const own =
      typeof input === "object" && input !== null
        ? Object.getOwnPropertyDescriptor(input, "__proto__")
        : undefined;
    return own !== undefined && own.enumerable === true && "value" in own
      ? { value: own.value }
      : undefined;
  }

  /**
   * The input as it was handed over, read before anything parses it.
   *
   * Which is the whole point for the three probes that rewrite themselves while they are being
   * read: what this schema owes is the record's answer about the value that ARRIVED, in the order
   * that value's keys were written in, and after a parse the object is no longer evidence of
   * either.
   */
  function asWritten(input: unknown): { keys: string[]; proto: { value: unknown } | undefined } {
    return {
      keys: typeof input === "object" && input !== null ? Object.keys(input) : [],
      proto: ownProto(input),
    };
  }

  /** What this schema owes for a value the record takes: the record's copy, plus that key. */
  function withProto(
    written: { keys: string[]; proto: { value: unknown } | undefined },
    copy: Record<string, unknown>,
  ): Record<string, unknown> {
    const own = written.proto;
    if (own === undefined) return copy;
    const ordered: Record<string, unknown> = {};
    for (const key of written.keys) {
      if (key === "__proto__") {
        Object.defineProperty(ordered, key, {
          value: own.value,
          writable: true,
          enumerable: true,
          configurable: true,
        });
      } else if (Object.hasOwn(copy, key)) {
        ordered[key] = copy[key];
      }
    }
    return ordered;
  }

  it("copies the way the record copies", () => {
    const value = { a: 1 };
    // Not the object that arrived: the copy is what keeps a hidden `toJSON` on it from deciding
    // what Galaxy is posted, and it is the record's copy rather than one invented here.
    expect(jsonObject().parse(value)).not.toBe(value);
    expect(jsonObject().parse(value)).toEqual(record.parse(value));
  });

  it("keeps a key the record parser drops, where the caller wrote it", () => {
    const text = '{"__proto__":{"direct":1},"keep":2}';
    const value = JSON.parse(text) as Record<string, unknown>;
    expect(Object.hasOwn(value, "__proto__")).toBe(true);
    expect(JSON.stringify(jsonObject().parse(value))).toBe(text);
    expect(JSON.stringify(record.parse(value))).toBe('{"keep":2}');
    // The key is a property and not a prototype, before and after.
    expect(Object.getPrototypeOf(jsonObject().parse(value))).toBe(Object.prototype);
  });

  it("takes the objects a record took, and refuses the rest in the same words", () => {
    const schema = jsonObject();
    expect(schema.safeParse({}).success).toBe(true);
    expect(schema.safeParse(Object.create(null) as Record<string, unknown>).success).toBe(true);
    for (const [value, received] of [
      ["x", "string"],
      [5, "number"],
      [true, "boolean"],
      [null, "null"],
      [undefined, "undefined"],
      [[1], "array"],
      [new Date(), "Date"],
      [new Map(), "Map"],
      [new (class Doc {})(), "Doc"],
    ] as Array<[unknown, string]>) {
      const ours = schema.safeParse(value);
      const theirs = record.safeParse(value);
      expect(ours.success, received).toBe(false);
      expect(theirs.success, received).toBe(false);
      expect(ours.error!.issues).toEqual(theirs.error!.issues);
      expect(ours.error!.issues[0]!.message).toBe(`Invalid input: expected record, received ${received}`);
    }
  });

  /**
   * Accept or refuse AND what comes back, value by value, against the record this replaces.
   *
   * The whole point of the table is that neither answer is re-derived here. Two earlier versions
   * of this schema derived one. The first asked a question of its own -- is the prototype
   * `Object.prototype` or null -- which reads like zod's and is not it, because zod asks whether
   * the value's constructor looks like `Object`; they part company on an object built in another
   * realm (taken there, refused here), on `Object.create(Object.create(null))` the same way
   * round, and on an own `constructor` key holding a function (refused there, taken here). The
   * second handed the arriving object back untouched, which let two things through that the
   * record's copy had stopped: a `toJSON` hidden on the value chose what was serialised, and a
   * symbol key -- which the record refuses, since it validates keys as well as filtering them --
   * was quietly dropped instead.
   *
   * So both halves are the record's, and this is the test that says so: the only difference
   * allowed is one own `__proto__` key put back where the caller wrote it.
   *
   * A probe is built rather than held, because the last three rewrite themselves while they are
   * being read and each side of the comparison has to be given one that has not been read yet.
   * The values whose identity the comparison can see -- a symbol key, a function, a Date -- are
   * made once above the table and handed to both.
   */
  it("takes, refuses and hands back exactly what a record does, plus that one key", () => {
    const inAnotherRealm = (expression: string): unknown =>
      runInNewContext(expression, { text: '{"a":1}' });
    const hidden = (define: PropertyDescriptor & { key: string }): Record<string, unknown> => {
      const { key, ...descriptor } = define;
      const value: Record<string, unknown> = { a: 1 };
      Object.defineProperty(value, key, descriptor);
      return value;
    };
    const getter = (): Record<string, unknown> => {
      const value: Record<string, unknown> = {};
      Object.defineProperty(value, "a", { get: () => 1, enumerable: true, configurable: true });
      return value;
    };
    const unexpected = Symbol("unexpected");
    const ctor = function ctor() {};
    const ownChoice = () => "own-choice";
    const when = new Date();
    /**
     * An enumerable getter that rewrites the object it is on, at the moment the record reads it.
     *
     * `note` is read and copied before `zap` runs, so the record's answer keeps it however the
     * getter leaves the input; the record's key list is taken once, before any of them run, so a
     * key that appears during the enumeration is not one it copied.
     */
    const rewritesItself = (text: string, rewrite: (value: Record<string, unknown>) => void) => () => {
      const value = JSON.parse(text) as Record<string, unknown>;
      Object.defineProperty(value, "zap", {
        get: () => {
          rewrite(value);
          return 1;
        },
        enumerable: true,
        configurable: true,
      });
      return value;
    };
    const defineProto = (value: Record<string, unknown>, held: unknown): void => {
      Object.defineProperty(value, "__proto__", {
        value: held,
        writable: true,
        enumerable: true,
        configurable: true,
      });
    };
    const probes: Array<[string, () => unknown]> = [
      ["{}", () => ({})],
      ["Object.create(null)", () => Object.create(null)],
      ["Object.create(Object.create(null))", () => Object.create(Object.create(null) as object)],
      ["a cross-realm object", () => inAnotherRealm("JSON.parse(text)")],
      ["a cross-realm Object.create(null)", () => inAnotherRealm("Object.create(null)")],
      ["an own constructor key holding a string", () => JSON.parse('{"constructor":"x"}')],
      ["an own constructor key holding an object", () => JSON.parse('{"constructor":{"prototype":{}}}')],
      ["an own constructor key holding a function", () => ({ constructor: ctor })],
      ["a frozen object", () => Object.freeze({ a: 1 })],
      ["a class instance", () => new (class Doc {})()],
      ["a Date", () => when],
      ["a Map", () => new Map()],
      ["an array", () => [1]],
      ["null", () => null],
      ["a string", () => "x"],
      ["a number", () => 5],
      ["a Proxy of {}", () => new Proxy({}, {})],
      ["JSON text with a __proto__ key", () => JSON.parse('{"__proto__":{"a":1}}')],
      // An own enumerable symbol key: the record refuses it as an `invalid_key`, and a schema
      // that took it would drop the key at serialisation without saying anything.
      ["an own enumerable symbol key", () => ({ a: 1, [unexpected]: 123 })],
      // A `toJSON` the record leaves behind, which is what makes the copy worth having.
      ["a non-enumerable own toJSON", () => hidden({ key: "toJSON", value: () => "wrong-document" })],
      ["a non-enumerable own data key", () => hidden({ key: "hush", value: 2 })],
      // And one written in the open, which is data like any other: the record copies it, so does
      // this, and `JSON.stringify` of either copy is whatever it returns.
      ["an enumerable own toJSON", () => ({ a: 1, toJSON: ownChoice })],
      ["an accessor property", () => getter()],
      ["an own __proto__ holding a primitive", () => JSON.parse('{"__proto__":"flat","b":2}')],
      ["JSON text with a __proto__ key and another", () => JSON.parse('{"__proto__":{"a":1},"b":2}')],
      // Not a key the record dropped by NAME -- it drops every non-enumerable own key -- so it
      // is not one to put back, and the caller's own `JSON.stringify` would not have shown it.
      ["a non-enumerable own __proto__", () => hidden({ key: "__proto__", value: { direct: 1 } })],
      // The three that rewrite themselves. A key the record copied stays copied, a key that
      // turns up too late for the record to have copied it is not in the answer, and an own
      // `__proto__` is whatever it was when the value arrived -- read once, like every other.
      [
        "a getter that deletes an earlier key",
        rewritesItself('{"__proto__":{"direct":1},"note":"keep","zap":0}', (value) => delete value.note),
      ],
      [
        "a getter that adds a key",
        rewritesItself('{"__proto__":{"direct":1},"note":"keep","zap":0}', (value) => {
          value.late = "added";
        }),
      ],
      [
        "a getter that adds an own __proto__",
        rewritesItself('{"note":"keep","zap":0}', (value) => defineProto(value, { late: 1 })),
      ],
      [
        "a getter that reassigns an own __proto__",
        rewritesItself('{"__proto__":{"written":1},"note":"keep","zap":0}', (value) =>
          defineProto(value, { swapped: 1 }),
        ),
      ],
    ];
    for (const [what, make] of probes) {
      const value = make();
      const written = asWritten(value);
      const ours = jsonObject().safeParse(value);
      // A value the record has not seen yet, so its answer is about what arrived.
      const theirs = record.safeParse(make());
      expect(ours.success, what).toBe(theirs.success);
      if (!ours.success) {
        expect(ours.error!.issues, what).toEqual(theirs.error!.issues);
        continue;
      }
      const expected = withProto(written, theirs.data as Record<string, unknown>);
      expect(ours.data, what).toEqual(expected);
      // Property by property: the attributes and the prototype are part of what the record
      // produced, so an added key has to look like one the record itself assigned.
      expect(Object.getOwnPropertyDescriptors(ours.data), what).toEqual(
        Object.getOwnPropertyDescriptors(expected),
      );
      expect(Object.getPrototypeOf(ours.data), what).toBe(Object.getPrototypeOf(expected));
      expect(JSON.stringify(ours.data), what).toBe(JSON.stringify(expected));
      // The record's own key set, plus `__proto__` where the caller wrote one and nothing else.
      const added = written.proto === undefined ? [] : ["__proto__"];
      expect([...Object.getOwnPropertyNames(ours.data)].sort(), what).toEqual(
        [...Object.getOwnPropertyNames(theirs.data), ...added].sort(),
      );
      expect(Object.getOwnPropertySymbols(ours.data), what).toEqual(
        Object.getOwnPropertySymbols(theirs.data),
      );
      // In the caller's order, so the JSON text Galaxy is posted is the text they wrote.
      expect(Reflect.ownKeys(ours.data), what).toEqual(Reflect.ownKeys(expected));
    }
  });

  it("leaves a hidden serialisation hook behind, and copies one written in the open", () => {
    const hidden: Record<string, unknown> = { class: "GalaxyUserTool" };
    Object.defineProperty(hidden, "toJSON", { value: () => "wrong-document" });
    // What the value would serialise as if it were handed on as it arrived.
    expect(JSON.stringify(hidden)).toBe('"wrong-document"');
    expect(JSON.stringify(jsonObject().parse(hidden))).toBe('{"class":"GalaxyUserTool"}');
    expect(JSON.stringify(record.parse(hidden))).toBe('{"class":"GalaxyUserTool"}');
    const open = { class: "GalaxyUserTool", toJSON: () => "own-choice" };
    expect(JSON.stringify(jsonObject().parse(open))).toBe('"own-choice"');
    expect(JSON.stringify(record.parse(open))).toBe('"own-choice"');
  });

  it("refuses a symbol key as the record's invalid_key", () => {
    const value = { class: "GalaxyUserTool", [Symbol("unexpected")]: 123 };
    const ours = jsonObject().safeParse(value);
    const theirs = record.safeParse(value);
    expect(ours.success).toBe(false);
    expect(ours.error!.issues).toEqual(theirs.error!.issues);
    expect(ours.error!.issues[0]!.code).toBe("invalid_key");
    expect(ours.error!.issues[0]!.message).toBe("Invalid key in record");
  });

  it("advertises what the record advertised, byte for byte", () => {
    const shape = (field: z.ZodType) => z.strictObject({ p: field, other: z.string() });
    const asJson = (field: z.ZodType) =>
      // The options the MCP SDK converts a tool's schema with.
      JSON.stringify(z.toJSONSchema(shape(field), { target: "draft-7", io: "input" }));
    expect(asJson(jsonObject())).toBe(asJson(record as z.ZodType));
    expect(asJson(jsonObject().describe("what it holds"))).toBe(
      asJson(record.describe("what it holds") as z.ZodType),
    );
    expect(asJson(jsonObject())).toContain(
      '"p":{"type":"object","propertyNames":{"type":"string"},"additionalProperties":{}}',
    );
    // Required, exactly as the record was, and a missing key still says so.
    expect(asJson(jsonObject())).toContain('"required":["p","other"]');
    expect(shape(jsonObject()).safeParse({ other: "x" }).error!.issues[0]!.message).toBe(
      "Invalid input: expected record, received undefined",
    );
  });

  it("is recognisable after the describe that clones it, and nothing else is", () => {
    expect(isJsonObjectSchema(jsonObject())).toBe(true);
    expect(isJsonObjectSchema(jsonObject().describe("what it holds"))).toBe(true);
    expect(isJsonObjectSchema(z.unknown())).toBe(false);
    expect(isJsonObjectSchema(z.string().describe("a string"))).toBe(false);
    expect(isJsonObjectSchema(record)).toBe(false);
    expect(isJsonObjectSchema(undefined)).toBe(false);
  });
});
