import { util, type ZodRawShape, type ZodTypeAny } from "zod";

/**
 * Accept the arguments the other surface accepts, before the schema sees them.
 *
 * FastMCP validates a tool call with pydantic in lax mode --
 * `type_adapter.validate_python(arguments)`, with no strict flag anywhere -- so on that
 * surface an integer parameter takes `"5"`, `" 5 "`, `"05"`, `"5.0"`, `1_000`, `true` and
 * `5.0`, and a boolean takes `"true"`, `"yes"`, `"off"`, `1` and `0`. The schemas here
 * declare exactly what Python's declare and nothing more, so without this a call that
 * works against one server is "expected number, received string" against the other.
 *
 * The leniency belongs at the same layer Python has it: where arguments arrive, not in the
 * contract. The schema an MCP client and the parity check are shown is untouched, which is
 * the whole point -- `type: integer` still means integer, and this is the decoding of an
 * argument into one, the way pydantic's non-strict mode is.
 *
 * Every rule below was measured against this repo's pydantic with
 * `TypeAdapter(int | None)` and `TypeAdapter(bool)`, in both Python and JSON mode, which
 * agree. Anything the tables do not accept is passed through untouched so the schema
 * refuses it and says why.
 */

/**
 * An object zod will go on to parse as a record: zod's own predicate, borrowed rather than
 * restated.
 *
 * `typeof` is not the question, because it says "object" for a Date, a Map, a Set, a boxed
 * `new Number(5)` and every class instance, and a transport that carries objects rather than JSON
 * text can deliver any of them. Spreading one of those produces `{}` -- an empty argument list,
 * which is a call that runs on its defaults -- where the SDK refuses the container outright; zod
 * turns all of them down (a plain object's `constructor`, if it has one, is a function whose
 * prototype has `isPrototypeOf` on it, and theirs is not), so that job is done.
 *
 * Nor is the prototype's identity the question, though it reads like it: `Object.prototype` or
 * null is one realm's answer, and an in-process caller on an `InMemoryTransport` can hand over an
 * object another realm parsed (`vm.runInNewContext("JSON.parse(text)", {text})`). Asking by
 * identity, one of those was neither decoded nor inspected here while the SDK's record parser went
 * on to accept it and drop a `__proto__` out of it -- the one thing this surface promises not to
 * do quietly. So the question is the SDK's parser's own (`zod/v4/core/schemas.js`, the record
 * path, gates on this same function), and what is decoded and what is inspected are exactly the
 * containers it will parse.
 */
export function isPlainObject(value: unknown): value is Record<string, unknown> {
  return util.isPlainObject(value);
}

/**
 * A caller's object, read once, as inert data.
 *
 * Everything on these surfaces that takes an object from a caller reads it HERE and nowhere else:
 * one read of its prototype, one `Reflect.ownKeys`, one descriptor per key, one read of each
 * value, onto a copy whose properties are all data. What comes back answers every later question
 * the same way it answered the first, which is the whole point. A transport that carries objects
 * rather than JSON text hands over LIVE ones -- a Proxy whose `ownKeys` reports something else the
 * second time it is asked, an accessor that returns a clean value twice and a `__proto__`-carrying
 * one on the third read, a `constructor` that answers `Object` and then `Date` -- so a surface
 * that reads such a value twice can inspect one object and go on to parse another. Five review
 * rounds found five of those, each a different pair of reads: a key snapshot and the record's own
 * enumeration, a dropped-key inspection and the SDK's parse, a decode and the SDK's parse, a
 * plainness check and the copy taken after it. pydantic reads a dict once. So this reads once, at
 * the boundary, and everything after it reads the copy.
 *
 * The contract, in one place, and no wider than the code. A caller sending JSON TEXT -- stdio,
 * HTTP, every ordinary client -- is unaffected by any of this, because text parses into an object
 * that says the same thing however often it is asked. An IN-PROCESS caller handing live objects
 * over an `InMemoryTransport` is promised this much: the eight fields the SDK's schemas declare
 * (`jsonrpc`, `id`, `method`, `params`; `name`, `arguments`, `_meta`, `task`), and every own
 * enumerable key of the message, of `params`, of an argument list and of each object-valued
 * parameter, are read exactly once, at the boundary, and what was read is what runs. A field that
 * came back `undefined` is asked `in` once besides, which separates absent from written as
 * undefined and runs no getter.
 *
 * And nothing wider. What this hands on rather than copies -- the value INSIDE one of those
 * fields, a property on a PROTOTYPE, a key JSON text cannot write -- belongs to whoever reads it
 * next, and is read as often as they read it: a getter on `_meta` runs three times for the SDK
 * on a call that succeeds, and one on `task` twice on a call it refuses. An object that answers
 * differently when asked again is answering them, not this.
 *
 * The copy keeps the ORIGINAL'S PROTOTYPE, because "is this a record?" is a question about the
 * prototype and it has to have the same answer before and after. zod's `isPlainObject` asks a
 * value for its `constructor`, and copying into a fresh `{}` answered that `Object` for a Date, a
 * Map and every class instance -- so the question had to be asked of the caller's object first,
 * which was a second read, and a `constructor` getter answering `Object` and then `Date` walked
 * through the gap between the two. With the prototype carried across, the question is the copy's
 * to answer: a Date is still not a record, an object another realm parsed still is, and an own
 * `constructor` is read once like any other key. What answers it is the prototype the value
 * really had and the own keys copied across with it, and not a trap's answer about either -- so a
 * Proxy around a class instance that says `Object` when it is asked for its `constructor` is
 * refused as `expected record, received Rep`, while an object whose PROTOTYPE says `Object` is
 * taken, because that is the prototype it has. Asked of the copy, the question can be asked more
 * than once at no risk, and is: the container gate and the decoding each ask it, so an inherited
 * `constructor` getter runs twice on a call this decodes.
 *
 * What it does not do is decide anything. An own enumerable key is copied with whatever it holds
 * -- a getter runs, once, and what it returned is stored as data, which is what zod's record does
 * with one too. A key that is not enumerable is carried by NAME alone, with no value: the record
 * and object parsers skip such a key while copying, so nothing downstream reads what it held, and
 * reading it here would run a getter neither parser would. `__proto__` is copied like any other
 * key, with `defineProperty` rather than assignment so that it stays a property instead of
 * becoming a prototype -- it is still the record's to drop by name and this surface's to refuse. A
 * symbol key is copied too, so the record can go on refusing an enumerable one as an
 * `invalid_key`; a non-enumerable symbol is carried by name like any other hidden key and then
 * seen by nobody, since the dropped-key count reads `getOwnPropertyNames` and the record skips
 * what is not enumerable. At the top of an argument list that is the one kind of key that can
 * arrive and disappear without a word; deeper in, where the refusal does not reach, a hidden key
 * of either sort goes the way the record has always taken it. JSON text can write neither.
 *
 * Only the top level. A nested value is handed on as it came, because that is what the record did
 * with one, and whatever parses that value reads it once at its own level if it comes back here
 * to do it -- `jsonObject()` does. `_meta` and `task` do not: what is inside those two is read by
 * the SDK's own schemas, as often as they ask for it.
 *
 * `except` names the keys the caller reads for itself -- `params.name` is read by property access,
 * since the SDK reads it that way and a peer may have put it on a prototype -- so that they are
 * left out here rather than read a second time.
 *
 * Anything that is not an object comes back as it came: there is nothing to read, and
 * `Reflect.ownKeys` of a primitive throws. So does an ARRAY, which is an object and is left alone
 * anyway -- no parser downstream reads the contents of one, both refuse it by type and name the
 * type in the refusal, and a copy carrying `Array.prototype` is not an array to `Array.isArray`,
 * so the only thing a copy would change is the word in a refusal that was going to happen either
 * way. Whether an object is one the parser downstream will TAKE is asked of what comes back from
 * here, never of what went in.
 */
export function materializeOnce(value: unknown, except?: readonly string[]): unknown {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return value;
  const from = value as Record<string | symbol, unknown>;
  const copy = Object.create(Object.getPrototypeOf(from) as object | null) as Record<string | symbol, unknown>;
  for (const key of Reflect.ownKeys(from)) {
    if (except !== undefined && typeof key === "string" && except.includes(key)) continue;
    const own = Object.getOwnPropertyDescriptor(from, key);
    // A Proxy may report a key from `ownKeys` and then no descriptor for it, and an earlier key's
    // getter may have deleted a later one. Either way there is nothing to copy.
    if (own === undefined) continue;
    const enumerable = own.enumerable === true;
    Object.defineProperty(copy, key, {
      value: enumerable ? from[key] : undefined,
      writable: true,
      enumerable,
      configurable: true,
    });
  }
  return copy;
}

const WRAPPERS = new Set(["optional", "nullable", "default", "nullish", "prefault"]);

/** Peel optional/nullable/default (and their stacks) off a field to reach what it really is. */
function unwrap(schema: ZodTypeAny): ZodTypeAny {
  const def = (schema as unknown as { def?: { type?: string; innerType?: ZodTypeAny } }).def;
  if (def?.type && WRAPPERS.has(def.type) && def.innerType) return unwrap(def.innerType);
  return schema;
}

type NumberDef = { type?: string; checks?: Array<{ _zod?: { def?: { format?: string; check?: string } } }> };

/** A `z.number().int()` field, which is what every numeric parameter here is. */
function isIntegerField(schema: ZodTypeAny): boolean {
  const inner = unwrap(schema) as unknown as { def?: NumberDef };
  if (inner.def?.type !== "number") return false;
  // zod records `.int()` as a format check on the number, so a plain float field is left alone.
  return (inner.def.checks ?? []).some(
    (c) => c._zod?.def?.format === "safeint" || c._zod?.def?.check === "number_format",
  );
}

function isBooleanField(schema: ZodTypeAny): boolean {
  return (unwrap(schema) as unknown as { def?: { type?: string } }).def?.type === "boolean";
}

/**
 * The whitespace pydantic strips around a number, which is not the whitespace `trim()`
 * strips. Every code point is compared by hand rather than through a character class,
 * because the string on the other side of this is whatever a caller sent.
 *
 * pydantic-core trims Rust's `char::is_whitespace`, which is Unicode White_Space. JavaScript
 * also strips U+FEFF, which pydantic does not -- so `"\uFEFF5"` is a number here and an
 * error there -- and JavaScript does not strip U+0085, which pydantic does, so `"\u00855"`
 * was the other way round. Every code point in this class was checked against the installed
 * pydantic with `TypeAdapter(int | None)` in both validate_python and validate_json; U+FEFF,
 * U+180E and U+200B were checked too and are refused there, so they are absent here.
 */
function isPydanticSpace(code: number): boolean {
  return (
    (code >= 0x09 && code <= 0x0d) ||
    code === 0x20 ||
    code === 0x85 ||
    code === 0xa0 ||
    code === 0x1680 ||
    (code >= 0x2000 && code <= 0x200a) ||
    code === 0x2028 ||
    code === 0x2029 ||
    code === 0x202f ||
    code === 0x205f ||
    code === 0x3000
  );
}

/**
 * The same trim, walked rather than matched.
 *
 * `^space+|space+$` with an anchored tail is quadratic: the engine tries the trailing branch
 * from every position, so `"5" + " ".repeat(64000) + "x"` -- which is not a number and was
 * always going to be refused -- took two seconds of one thread before zod ever saw it. Two
 * scans and one slice is linear, and the same input is under a millisecond.
 */
function trimPydanticSpace(value: string): string {
  let start = 0;
  let end = value.length;
  while (start < end && isPydanticSpace(value.charCodeAt(start))) start += 1;
  while (end > start && isPydanticSpace(value.charCodeAt(end - 1))) end -= 1;
  return start === 0 && end === value.length ? value : value.slice(start, end);
}

/**
 * pydantic's string-to-int grammar, which is narrower than JavaScript's `Number`.
 *
 * A sign is allowed; digits may be grouped by single underscores; a fractional part is
 * allowed only when it is all zeros. No exponent, no hex, no thousands separators, no
 * non-ASCII digits -- `"1e2"`, `"0x10"`, `"1,000"`, `"٥"` and `"5.5"` are all refused there,
 * and `Number()` would have taken most of them.
 */
const INT_STRING = /^[+-]?(?:\d(?:_?\d)*)(?:\.0+)?$/;

/** What pydantic makes of `value` for an integer parameter, or `undefined` for "not an integer". */
function asInteger(value: unknown): number | undefined {
  if (typeof value === "number") {
    // An integral float is an integer there (`5.0` -> 5, `-0.0` -> 0); a fraction is not.
    return Number.isFinite(value) && Number.isInteger(value) ? value + 0 : undefined;
  }
  // `true` is 1 and `false` is 0, which pydantic allows for an int and JavaScript would not.
  if (typeof value === "boolean") return value ? 1 : 0;
  if (typeof value !== "string") return undefined;
  const text = trimPydanticSpace(value);
  if (!INT_STRING.test(text)) return undefined;
  const parsed = Number(text.replace(/_/g, ""));
  // Beyond 2^53 pydantic keeps the exact integer and JavaScript cannot, so rather than hand
  // back a silently different number this leaves the string for the schema to refuse.
  return Number.isSafeInteger(parsed) ? parsed : undefined;
}

const TRUE_WORDS = new Set(["true", "t", "yes", "y", "on", "1"]);
const FALSE_WORDS = new Set(["false", "f", "no", "n", "off", "0"]);

/** What pydantic makes of `value` for a boolean parameter, or `undefined` for "not a boolean". */
function asBoolean(value: unknown): boolean | undefined {
  if (typeof value === "boolean") return value;
  // Exactly 1 and 0, including 1.0 and 0.0; 2 is refused there and is refused here.
  if (typeof value === "number") return value === 1 ? true : value === 0 ? false : undefined;
  if (typeof value !== "string") return undefined;
  // No surrounding whitespace: `" true "` is refused there too.
  const word = value.toLowerCase();
  if (TRUE_WORDS.has(word)) return true;
  if (FALSE_WORDS.has(word)) return false;
  return undefined;
}

/**
 * A copy of `raw` with the integer and boolean arguments decoded the way pydantic decodes
 * them. Keys the shape does not declare, and values neither table accepts, are passed
 * through untouched.
 */
export function laxenLikePydantic(shape: ZodRawShape, raw: Record<string, unknown>): Record<string, unknown> {
  // Anything that is not a plain object is handed back as it came: spreading a Date or a Map
  // would turn a malformed argument list into an empty one, which is a different call.
  if (!isPlainObject(raw)) return raw;
  const out: Record<string, unknown> = { ...raw };
  for (const [key, schema] of Object.entries(shape)) {
    if (!(key in out)) continue;
    const value = out[key];
    // null and undefined mean "not given" or "explicitly null"; the schema decides whether
    // either is allowed, exactly as pydantic does for `int | None`.
    if (value === null || value === undefined) continue;
    if (isIntegerField(schema as ZodTypeAny)) {
      const asInt = asInteger(value);
      if (asInt !== undefined) out[key] = asInt;
    } else if (isBooleanField(schema as ZodTypeAny)) {
      const asBool = asBoolean(value);
      if (asBool !== undefined) out[key] = asBool;
    }
  }
  return out;
}
