import { readFileSync } from "node:fs";
import { z, type ZodRawShape, type ZodTypeAny } from "zod";
import { laxenLikePydantic } from "@galaxyproject/galaxy-ops";

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

/**
 * Reassemble the op input object from commander positionals + options, then validate.
 *
 * Commander hands every argument over as a string, and the schemas take a real number or a
 * real boolean and nothing else -- `--limit 5` would otherwise be "expected number, received
 * string". The decoding is `laxenLikePydantic`, the same one the MCP surface applies to an
 * incoming call and the same one FastMCP gets from pydantic's non-strict mode, so `--limit 5`,
 * `--limit 5.0` and `--active yes` mean here what they mean there. Anything the tables do not
 * accept is passed through untouched, so the schema refuses it and says why: `--limit abc`
 * stays "abc" rather than arriving as NaN.
 */
export function buildInput(shape: ZodRawShape, positionals: string[], options: Record<string, unknown>) {
  const raw: Record<string, unknown> = {};
  let pi = 0;
  for (const [key, schema] of Object.entries(shape)) {
    const kind = classifyField(schema as ZodTypeAny);
    if (kind === "positional") {
      if (pi < positionals.length) raw[key] = positionals[pi++];
    } else {
      const flag = flagName(key);
      const val = options[key] ?? options[flag.replace(/-([a-z])/g, (_, c) => c.toUpperCase())];
      if (val === undefined) continue;
      raw[key] = kind === "json" ? readJsonArg(String(val)) : val;
    }
  }
  return z.object(shape).safeParse(laxenLikePydantic(shape, raw));
}
