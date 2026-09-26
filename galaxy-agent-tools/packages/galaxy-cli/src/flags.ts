import { readFileSync } from "node:fs";
import { z, type ZodRawShape, type ZodTypeAny } from "zod";

export type FieldKind = "positional" | "option" | "boolean" | "json";

const WRAPPERS = new Set(["optional", "nullable", "default"]);

function unwrap(schema: ZodTypeAny): ZodTypeAny {
  // Zod v4 exposes the def under `.def`; optional/nullable/default each wrap an innerType, and
  // .nullish() stacks two of them, so this has to keep peeling rather than peel once.
  const def = (schema as unknown as { def?: { type?: string; innerType?: ZodTypeAny } }).def;
  if (def && def.type && WRAPPERS.has(def.type) && def.innerType) return unwrap(def.innerType);
  return schema;
}
function typeTag(schema: ZodTypeAny): string | undefined {
  return (unwrap(schema) as unknown as { def?: { type?: string } }).def?.type;
}
function unionMembers(schema: ZodTypeAny): ZodTypeAny[] | undefined {
  const def = (unwrap(schema) as unknown as { def?: { type?: string; options?: ZodTypeAny[] } }).def;
  return def?.type === "union" ? def.options : undefined;
}
function isOptional(schema: ZodTypeAny): boolean {
  return schema.safeParse(undefined).success;
}

const isJsonTag = (tag: string | undefined) => tag === "record" || tag === "object";

export function classifyField(schema: ZodTypeAny): FieldKind {
  const tag = typeTag(schema);
  if (isJsonTag(tag)) return "json";
  // A field that takes an object OR the JSON string of one is still a JSON flag here, or
  // `--inputs @file.json` stops reading the file and hands the op the literal "@file.json".
  if (unionMembers(schema)?.some((m) => isJsonTag(typeTag(m)))) return "json";
  if (tag === "boolean") return "boolean";
  return isOptional(schema) ? "option" : "positional";
}

/** camelCase -> kebab-case for flag names. */
export function flagName(key: string): string {
  return key.replace(/[A-Z]/g, (m) => "-" + m.toLowerCase());
}

function readJsonArg(raw: string): unknown {
  const text = raw.startsWith("@") ? readFileSync(raw.slice(1), "utf8") : raw;
  return JSON.parse(text);
}

/** True when the field wants a number, so an argument off the command line needs converting. */
export function isNumberField(schema: ZodTypeAny): boolean {
  return typeTag(schema) === "number";
}

/**
 * Commander hands every argument over as a string, and the op schemas want a real number --
 * `--limit 5` is otherwise "expected number, received string". Anything that is not a number
 * is left alone rather than mangled, so the schema gets to refuse it and say why: "" stays ""
 * instead of becoming 0, and "abc" stays "abc" instead of becoming NaN.
 */
function readNumberArg(raw: unknown): unknown {
  if (typeof raw !== "string" || raw.trim() === "") return raw;
  const n = Number(raw);
  return Number.isNaN(n) ? raw : n;
}

/** Reassemble the op input object from commander positionals + options, then validate. */
export function buildInput(shape: ZodRawShape, positionals: string[], options: Record<string, unknown>) {
  const raw: Record<string, unknown> = {};
  let pi = 0;
  for (const [key, schema] of Object.entries(shape)) {
    const kind = classifyField(schema as ZodTypeAny);
    const wantsNumber = isNumberField(schema as ZodTypeAny);
    if (kind === "positional") {
      if (pi < positionals.length) {
        const val = positionals[pi++];
        raw[key] = wantsNumber ? readNumberArg(val) : val;
      }
    } else {
      const flag = flagName(key);
      const val = options[key] ?? options[flag.replace(/-([a-z])/g, (_, c) => c.toUpperCase())];
      if (val === undefined) continue;
      if (kind === "json") raw[key] = readJsonArg(String(val));
      else raw[key] = wantsNumber ? readNumberArg(val) : val;
    }
  }
  return z.object(shape).safeParse(raw);
}
