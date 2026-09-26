import { z, type ZodType } from "zod";
import { materializeOnce } from "./laxen";

/**
 * The record these five parameters were declared as, kept on to do their parsing.
 *
 * One is enough, and it can be shared: it parses a payload handed to it and holds nothing of
 * its own between calls.
 */
const record = z.record(z.string(), z.unknown());

/**
 * A parameter that takes the caller's own JSON object, `__proto__` and all.
 *
 * `z.record(z.string(), z.unknown())` was the obvious way to say "an object of anything", and it
 * is nearly the right one. It parses by copying the object into a fresh `{}`, and while copying
 * it skips a key named `__proto__` unconditionally (`zod/v4/core/schemas.js`, the record path;
 * its object parser does the same in `handleCatchall`). So a `representation` or a workflow's
 * `inputs` written with `"__proto__": {...}` reached Galaxy without it: a call that succeeded and
 * posted a document its caller had not written. The Python server takes these parameters as
 * `dict[str, Any]` and pydantic keeps the key, so the two surfaces disagreed about the caller's
 * own data -- which both of them promise to pass on exactly as written.
 *
 * So the record still does the parsing here, and this adds back the one key it drops. What comes
 * out is the record's own output -- its acceptance, its issues, its copy -- and then, if the
 * caller wrote an own `__proto__`, that key defined on the copy as a data property. Everything
 * else the record does to a value is its business and is left alone: an own property that is not
 * enumerable is dropped (a `toJSON` hidden that way does not get to decide what Galaxy is
 * posted), an enumerable one is copied whatever it holds (a `toJSON` written in the open does),
 * a getter is read once and stored as data, a symbol key is refused as an `invalid_key`, and a
 * frozen object comes back unfrozen.
 *
 * Deciding any of that here instead is what went wrong twice, and reading the caller's object more
 * than once was the rest. A prototype-identity test (the prototype is `Object.prototype` or null)
 * reads like zod's `util.isPlainObject` and is not it, so an object parsed in ANOTHER REALM --
 * which an in-process caller on an `InMemoryTransport` can really hand over, as
 * `vm.runInNewContext("JSON.parse(text)", {text})` -- was refused. Then handing the arriving
 * object straight back looked like the smaller change and was the larger one: with no copy in the
 * way, a non-enumerable `toJSON` on it chose what was posted. Then the key order was read off the
 * caller's object a second time, after the record had finished, and an enumerable getter can
 * delete a key while the record is reading the one before it -- so a key the record had already
 * copied was missing from the body that went out. Taking the order first fixed that one pair of
 * reads and left the rest: the order came off one `Reflect.ownKeys` and the record then ran its
 * own, and a Proxy that reports a shorter key list the second time it is asked lost a field out of
 * the posted document with no `__proto__` anywhere near it.
 *
 * So `materializeOnce` reads the caller's object exactly once, at the top of the check, into inert
 * data -- and everything after that line reads the copy: the record's enumeration, its values, its
 * issues (error finalisation reads `issue.input`, the way it does for any record, and what it
 * finds there is the copy or a key off it), the `__proto__` put back, and the order it goes back
 * in. The caller's object is read once and never again.
 *
 * Including the question of whether it is a record at all, which used to be asked here first and
 * of the caller's object, because a copy into a fresh `{}` was a plain object whatever it had been
 * copied from. That was the last pair of reads: a `constructor` getter answering `Object` and then
 * `Date` passed the check and was stored as `Date`, so a representation main accepted was refused
 * -- and the other way round is the same gap. The copy carries the original's prototype now, so
 * the question is the record's own and it asks it of the copy: a Date, a Map and a class instance
 * are still refused by type and still named by type in the refusal, an object another realm parsed
 * is still taken, and a `constructor` a caller wrote as an own key is read once like any other.
 *
 * What a client is shown does not change: the metadata below is the JSON Schema the record
 * advertised, byte for byte, and the parity check compares that against the Python manifest. Nor
 * does the refusal, down to the word -- `expected record` reads oddly for a schema that is no
 * longer one, but a caller reading "expected record, received string" today would have to be
 * told something new for no reason.
 */
export function jsonObject(): ZodType<Record<string, unknown>, Record<string, unknown>> {
  return (
    z
      .unknown()
      .check((c) => {
        // One read, before anything asks the value anything -- including whether it is a record,
        // which is the record's own question below and is asked of what comes back from here. A
        // primitive and an array come back as they came; everything else comes back as data on
        // the prototype it arrived with, so the type the record refuses and names is the type
        // that arrived.
        const inert = materializeOnce(c.value);
        const parsed = record._zod.run({ value: inert, issues: [] }, {});
        if (parsed instanceof Promise) {
          // Unreachable: `z.string()` keys and `z.unknown()` values both parse synchronously,
          // and zod's record parser throws rather than await a key schema in the first place.
          throw new Error("jsonObject: the record parsed asynchronously");
        }
        if (parsed.issues.length > 0) {
          // Raw as the record raised them, so the outer parse finalises them -- message, path
          // and any error map of its own -- exactly as it would have finalised a record's.
          c.issues.push(...parsed.issues);
          return;
        }
        // The record took it, so the value it took is the copy, never the caller's object.
        c.value = withOwnProto(inert as Record<string, unknown>, parsed.value as Record<string, unknown>);
      })
      // Freshly built each time, because what goes in here is assigned into the emitted JSON
      // Schema by reference rather than copied.
      .meta({ type: "object", propertyNames: { type: "string" }, additionalProperties: {} })
  ) as unknown as ZodType<Record<string, unknown>, Record<string, unknown>>;
}

/**
 * The record's copy, with the key it skipped by name put back where the caller wrote it.
 *
 * Both halves are data this realm built: `copy` is what the record handed back, and `inert` is the
 * caller's object as it was read at the boundary -- same keys in the same order, every one of them
 * a data property. So the questions asked here have one answer each however strange the value that
 * arrived was.
 *
 * The key is defined rather than assigned, since assigning that name sets the copy's prototype
 * instead of writing a property. The copy is then rebuilt in the order the keys were written in,
 * so the JSON text Galaxy is posted is the caller's text key for key -- and the values come off
 * the record's copy alone, so which keys there are and what they hold is still its answer. A key
 * the order has and the copy does not is simply absent: the record decided that, for a name it
 * skips or a property it was not shown. A key the copy has and the order does not goes on the end,
 * so nothing the record copied can go missing here.
 *
 * Own, enumerable and holding a value is what the record asks of every key it does copy, so the
 * `__proto__` put back is exactly the one it turned down for being called that -- a non-enumerable
 * one is not a key the caller's own `JSON.stringify` would have shown either, and is left where
 * the record left it.
 */
function withOwnProto(inert: Record<string, unknown>, copy: Record<string, unknown>): Record<string, unknown> {
  const own = Object.getOwnPropertyDescriptor(inert, "__proto__");
  if (own === undefined || own.enumerable !== true) return copy;
  const ordered: Record<string, unknown> = {};
  for (const key of Reflect.ownKeys(inert)) {
    // Unreachable: a symbol key is the record's `invalid_key` above, never a copied one.
    if (typeof key !== "string") continue;
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
  for (const key of Object.keys(copy)) {
    if (!Object.hasOwn(ordered, key)) ordered[key] = copy[key];
  }
  return ordered;
}

/**
 * Whether a field is one of the above -- asked of the metadata rather than of the def,
 * because `z.unknown()` is what the def says and "an open object" is what the field means.
 * Metadata is also what survives `.describe()`, which clones the schema.
 */
export function isJsonObjectSchema(schema: unknown): boolean {
  // Asked of anything, because the callers asking are walking a shape: the registry reads
  // `_zod` off what it is handed and a field that turned out not to be a schema is a `false`
  // here rather than a TypeError somewhere else.
  if (typeof schema !== "object" || schema === null || !("_zod" in schema)) return false;
  const meta = z.globalRegistry.get(schema as ZodType) as
    | { type?: unknown; propertyNames?: { type?: unknown } }
    | undefined;
  return meta?.type === "object" && meta.propertyNames?.type === "string";
}
