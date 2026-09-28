/**
 * Reading a value out of a Galaxy record the way Python reads it, and rendering it the
 * way an f-string renders it.
 *
 * Both belong to the other server's summary lines. Almost every one of them names
 * something it read out of a record -- a history's name, a page's id, a collection's
 * name -- with `record.get(key, fallback)` and drops it straight into an f-string, and
 * neither of those two steps is what the JavaScript spelling beside it does.
 */
import { pyRepr } from "./python-str";

/**
 * `dict.get(key, fallback)`: the fallback stands in for an ABSENT key and for nothing
 * else, so a field Galaxy sent as null comes back null rather than defaulted.
 *
 * `record[key] ?? fallback` is the line this replaces, and it differs on exactly the
 * records that matter: a collection whose `name` Galaxy states as null reads as null
 * here and as the fallback there, and the two servers then name it differently.
 */
export function pyGet(record: Record<string, unknown>, key: string, fallback: unknown): unknown {
  return Object.prototype.hasOwnProperty.call(record, key) ? record[key] : fallback;
}

/** `repr()` of a value, which is what `str()` of a list or a dict renders its insides with. */
function pyReprValue(value: unknown): string {
  if (typeof value === "string") return pyRepr(value);
  if (Array.isArray(value)) return `[${value.map(pyReprValue).join(", ")}]`;
  if (typeof value === "object" && value !== null) {
    const entries = Object.entries(value as Record<string, unknown>);
    return `{${entries.map(([k, v]) => `${pyRepr(k)}: ${pyReprValue(v)}`).join(", ")}}`;
  }
  return pyStr(value);
}

/**
 * What `f"{value}"` renders, which is `str(value)` and not `repr(value)` -- so a string
 * arrives unquoted and the quotes in the sentences here are the f-string's own.
 *
 * The cases that reach these sentences are a string and a null, because that is what
 * Galaxy states a name, a title or an id as, and those two are exact: null renders
 * `None`, which is what an agent reading the other server's answer sees for a nameless
 * collection. A JavaScript `undefined`, which no JSON document carries, stands in for
 * the same absent value and renders the same way. `True`, `False`, an integer, and a
 * list or an object rendered through `repr()` are exact as well.
 *
 * One spelling is NOT exact and cannot be made so from a parsed JSON value: a
 * non-integer number, because `JSON.parse` loses the difference between Python's `int`
 * and `float` -- `1.0` arrives as the value `1` does, and Python renders the two as
 * `1.0` and `1`. A whole value renders as an integer here, which is the common case and
 * is what Galaxy sends for a count.
 */
export function pyStr(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "None";
  if (value === true) return "True";
  if (value === false) return "False";
  if (typeof value === "object") return pyReprValue(value);
  return String(value);
}
