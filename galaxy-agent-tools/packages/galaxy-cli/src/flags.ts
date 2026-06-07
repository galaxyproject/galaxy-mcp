import { readFileSync } from "node:fs";
import { z, type ZodRawShape, type ZodTypeAny } from "zod";

export type FieldKind = "positional" | "option" | "boolean" | "json";

function unwrap(schema: ZodTypeAny): ZodTypeAny {
  // Zod v4 exposes the def under `.def`; optional/default wrap an innerType.
  const def = (schema as unknown as { def?: { type?: string; innerType?: ZodTypeAny } }).def;
  if (def && (def.type === "optional" || def.type === "default") && def.innerType) return unwrap(def.innerType);
  return schema;
}
function typeTag(schema: ZodTypeAny): string | undefined {
  return (unwrap(schema) as unknown as { def?: { type?: string } }).def?.type;
}
function isOptional(schema: ZodTypeAny): boolean {
  return schema.safeParse(undefined).success;
}

export function classifyField(schema: ZodTypeAny): FieldKind {
  const tag = typeTag(schema);
  if (tag === "record" || tag === "object") return "json";
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

/** Reassemble the op input object from commander positionals + options, then validate. */
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
  return z.object(shape).safeParse(raw);
}
