import { readFileSync } from "node:fs";
import { z, type ZodRawShape, type ZodTypeAny } from "zod";
import { isJsonObjectSchema, laxenLikePydantic, spellParamNames } from "@galaxyproject/galaxy-ops";

export type FieldKind = "positional" | "option" | "boolean" | "json" | "array";

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

/**
 * A field whose value is a JSON object, whichever way the op said so: a `z.object`, a
 * `z.record`, or a `jsonObject()` -- which is a `z.unknown()` underneath, so its def cannot
 * be asked and `isJsonObjectSchema` reads the object it advertises instead.
 */
const takesJson = (schema: ZodTypeAny): boolean => {
  const tag = typeTag(schema);
  return tag === "record" || tag === "object" || isJsonObjectSchema(unwrap(schema));
};

export function classifyField(schema: ZodTypeAny): FieldKind {
  const tag = typeTag(schema);
  if (takesJson(schema)) return "json";
  // A field that takes an object OR the JSON string of one is still a JSON flag here, or
  // `--inputs @file.json` stops reading the file and hands the op the literal "@file.json".
  if (unionMembers(schema)?.some(takesJson)) return "json";
  if (tag === "boolean") return "boolean";
  // A list is a repeatable flag whether it is required or not. As a positional it could only
  // ever arrive as one string, which every array schema here refuses -- so an op that takes
  // one had no working command line at all.
  if (tag === "array") return "array";
  return isOptional(schema) ? "option" : "positional";
}

/**
 * Optional fields that stay a positional argument because they were a required one.
 *
 * `galaxy-cli get_job_details <datasetId>` is what every script written against 0.3.0
 * runs, and 0.3.1 made the field optional so a job can be asked for by `--job-id` instead.
 * By the rule in `classifyField` an optional field is a flag, which would turn that command
 * line into a usage error over a patch release. So the field is both: `[datasetId]` as
 * before, and `--dataset-id` like every other optional; `buildInput` refuses the two
 * together rather than letting one quietly win.
 */
const KEPT_POSITIONALS: Readonly<Record<string, readonly string[]>> = {
  get_job_details: ["datasetId"],
};

/** The fields of an op that are taken as an optional positional as well as a flag. */
export function keptPositionals(opName: string): readonly string[] {
  return KEPT_POSITIONALS[opName] ?? [];
}

/** camelCase -> kebab-case for flag names. */
export function flagName(key: string): string {
  return key.replace(/[A-Z]/g, (m) => "-" + m.toLowerCase());
}

/** What a caller writes on the command line for one of an op's inputs. */
export function cliParamName(input: ZodRawShape, key: string): string {
  const schema = input[key] as ZodTypeAny | undefined;
  if (schema === undefined) return key;
  return classifyField(schema) === "positional" ? `<${key}>` : `--${flagName(key)}`;
}

/**
 * Help text with the op's parameter names spelled the way this CLI takes them.
 *
 * A description that says "pass sectionId" names nothing a command line accepts; the flag is
 * `--section-id`, and over MCP the same sentence says `section_id`. One sentence, respelled
 * per surface -- see `spellParamNames` for why only the camelCase keys move.
 */
export function inCliNames(text: string, input: ZodRawShape): string {
  return spellParamNames(text, input, (key) => cliParamName(input, key));
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
export type BuiltInput =
  | { success: true; data: Record<string, unknown> }
  | { success: false; error: { message: string } };

export function buildInput(
  shape: ZodRawShape,
  positionals: (string | undefined)[],
  options: Record<string, unknown>,
  kept: readonly string[] = [],
): BuiltInput {
  const raw: Record<string, unknown> = {};
  let pi = 0;
  for (const [key, schema] of Object.entries(shape)) {
    const kind = classifyField(schema as ZodTypeAny);
    const isKept = kept.includes(key);
    if (kind === "positional" || isKept) {
      // Commander hands an optional argument that was left out over as undefined.
      const value = pi < positionals.length ? positionals[pi++] : undefined;
      if (value !== undefined) raw[key] = value;
      if (!isKept) continue;
    }
    const flag = flagName(key);
    const val = options[key] ?? options[flag.replace(/-([a-z])/g, (_, c) => c.toUpperCase())];
    if (val === undefined) continue;
    if (key in raw) {
      return {
        success: false,
        error: { message: `${key} was given twice, as the positional argument and as --${flag}; give it once` },
      };
    }
    raw[key] = kind === "json" ? readJsonArg(String(val)) : val;
  }
  const parsed = z.object(shape).safeParse(laxenLikePydantic(shape, raw));
  return parsed.success
    ? { success: true, data: parsed.data as Record<string, unknown> }
    : { success: false, error: { message: parsed.error.message } };
}
