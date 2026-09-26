import type { ZodRawShape, ZodTypeAny } from "zod";

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
 * A plain JSON object: `{}` or `Object.create(null)`, and nothing else.
 *
 * Decided by prototype rather than by `typeof`, because `typeof` says "object" for a Date, a
 * Map, a Set, a boxed `new Number(5)` and every class instance, and a transport that carries
 * objects rather than JSON text can deliver any of them. Spreading one of those produces
 * `{}` -- an empty argument list, which is a call that runs on its defaults -- where the SDK
 * would have refused the container outright. One definition, used by the decoder and by the
 * surface that hands arguments to it.
 */
export function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null) return false;
  const proto = Object.getPrototypeOf(value) as object | null;
  return proto === Object.prototype || proto === null;
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
