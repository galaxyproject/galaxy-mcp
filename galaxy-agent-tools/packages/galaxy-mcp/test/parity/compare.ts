/**
 * Contract comparison between the Python MCP server's generated surface manifest
 * (mcp-server-galaxy-py/tests/testdata/mcp-surface.json) and the tools this
 * package advertises.
 *
 * Both sides describe themselves as JSON Schema, so the comparison is over
 * parameter names, types, requiredness and declared defaults, plus whether each
 * tool says it mutates anything. Descriptions are deliberately not compared: the
 * two surfaces word things differently on purpose and diffing prose would bury
 * the contract differences that matter.
 */

export type DivergenceKind =
  | "missing-ts-tool"
  | "missing-py-tool"
  | "missing-ts-param"
  | "missing-py-param"
  | "type-mismatch"
  | "required-mismatch"
  | "default-mismatch"
  | "mutability-mismatch";

/** Kinds that are about a whole tool rather than one of its parameters. */
export const WHOLE_TOOL_KINDS: readonly DivergenceKind[] = [
  "missing-ts-tool",
  "missing-py-tool",
  "mutability-mismatch",
];

export interface Divergence {
  tool: string;
  /** null for whole-tool divergences. */
  param: string | null;
  kind: DivergenceKind;
  /** What each side actually says, so an accepted entry goes stale when either moves. */
  observed: string;
}

export interface JsonSchema {
  type?: string;
  anyOf?: JsonSchema[];
  items?: JsonSchema;
  enum?: unknown[];
  properties?: Record<string, JsonSchema>;
  required?: string[];
  default?: unknown;
}

/** One tool as its own surface advertises it. */
export interface ToolContract {
  inputSchema: JsonSchema;
  /** The tool says it changes something: Python's `write` tag, MCP's `readOnlyHint: false`. */
  mutating: boolean;
}

/**
 * Whole-surface differences that would otherwise produce a divergence per
 * parameter. Every rule is a single switch, flipped from the registry, and adding
 * one here forces it to be declared there.
 */
export const NORMALIZATION_RULES = [
  "snakeCaseParamNames",
  "pythonNullDefaults",
  "optionalNullUnions",
] as const;

export type NormalizationRuleName = (typeof NORMALIZATION_RULES)[number];
export type Normalization = Record<NormalizationRuleName, boolean>;

export interface NormalParam {
  type: string;
  required: boolean;
  hasDefault: boolean;
  default?: unknown;
}

export function toSnakeCase(name: string): string {
  return name.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase();
}

function isNullable(schema: JsonSchema): boolean {
  return (schema.anyOf ?? []).some((v) => v.type === "null");
}

function withoutNull(schema: JsonSchema): JsonSchema {
  if (!schema.anyOf) return schema;
  const variants = schema.anyOf.filter((v) => v.type !== "null");
  const [only] = variants;
  return only && variants.length === 1 ? only : { ...schema, anyOf: variants };
}

/** JSON with keys in a fixed order and prose dropped, so two shapes compare by structure. */
function stableStringify(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  if (value && typeof value === "object") {
    const fields = Object.entries(value as Record<string, unknown>)
      .filter(([key]) => key !== "description")
      .sort(([a], [b]) => (a < b ? -1 : 1))
      .map(([key, v]) => `${JSON.stringify(key)}:${stableStringify(v)}`);
    return `{${fields.join(",")}}`;
  }
  return JSON.stringify(value) ?? "null";
}

/**
 * A comparable shorthand for a parameter's type, e.g. `string`, `array<string>`.
 *
 * It models the constructs both surfaces actually use. It does NOT compare
 * value constraints (format, minimum, pattern), `additionalProperties`, or the
 * inner shape of an object -- an open record and a closed one both read `object`.
 * Anything it does not model at all falls through to a structural comparison
 * rather than to a single catch-all token, so two unlike shapes cannot match by
 * accident.
 */
export function typeToken(schema: JsonSchema): string {
  if (schema.enum) {
    return `enum<${schema.enum.map((v) => JSON.stringify(v)).sort().join("|")}>`;
  }
  if (schema.anyOf) return `anyOf<${schema.anyOf.map(typeToken).sort().join("|")}>`;
  if (schema.type === "array") return `array<${typeToken(schema.items ?? {})}>`;
  if (schema.type) return schema.type;
  return `schema(${stableStringify(schema)})`;
}

export function normalizeParams(
  schema: JsonSchema,
  rules: Normalization,
  where: string,
): Map<string, NormalParam> {
  const required = new Set(schema.required ?? []);
  const out = new Map<string, NormalParam>();
  for (const [rawName, prop] of Object.entries(schema.properties ?? {})) {
    const isRequired = required.has(rawName);
    const optionalNull = !isRequired && isNullable(prop);
    const typeSchema = rules.optionalNullUnions && optionalNull ? withoutNull(prop) : prop;
    const hasDefault =
      "default" in prop && !(rules.pythonNullDefaults && optionalNull && prop.default === null);
    const name = rules.snakeCaseParamNames ? toSnakeCase(rawName) : rawName;
    if (out.has(name)) {
      throw new Error(
        `${where}: "${rawName}" normalizes to "${name}", which another parameter already ` +
          "uses, so the comparison would silently drop one of them",
      );
    }
    out.set(name, {
      type: typeToken(typeSchema),
      required: isRequired,
      hasDefault,
      ...(hasDefault ? { default: prop.default } : {}),
    });
  }
  return out;
}

const show = (p: NormalParam): string => (p.hasDefault ? JSON.stringify(p.default) : "none");

function compareTool(
  tool: string,
  python: ToolContract,
  typescript: ToolContract,
  rules: Normalization,
): Divergence[] {
  const found: Divergence[] = [];
  if (python.mutating !== typescript.mutating) {
    found.push({
      tool,
      param: null,
      kind: "mutability-mismatch",
      observed: `python=${python.mutating ? "write" : "read"} typescript=${
        typescript.mutating ? "write" : "read"
      }`,
    });
  }
  const py = normalizeParams(python.inputSchema, rules, `${tool} (python)`);
  const ts = normalizeParams(typescript.inputSchema, rules, `${tool} (typescript)`);
  for (const param of py.keys()) {
    if (!ts.has(param)) found.push({ tool, param, kind: "missing-ts-param", observed: "" });
  }
  for (const param of ts.keys()) {
    if (!py.has(param)) found.push({ tool, param, kind: "missing-py-param", observed: "" });
  }
  for (const [param, p] of py) {
    const t = ts.get(param);
    if (!t) continue;
    if (p.type !== t.type) {
      found.push({
        tool,
        param,
        kind: "type-mismatch",
        observed: `python=${p.type} typescript=${t.type}`,
      });
    }
    if (p.required !== t.required) {
      found.push({
        tool,
        param,
        kind: "required-mismatch",
        observed: `python=${p.required} typescript=${t.required}`,
      });
    }
    if (p.hasDefault !== t.hasDefault || JSON.stringify(p.default) !== JSON.stringify(t.default)) {
      found.push({
        tool,
        param,
        kind: "default-mismatch",
        observed: `python=${show(p)} typescript=${show(t)}`,
      });
    }
  }
  return found;
}

/** Every way the two surfaces disagree, in a stable order. */
export function compareSurfaces(
  python: Record<string, ToolContract>,
  typescript: Record<string, ToolContract>,
  rules: Normalization,
): Divergence[] {
  const found: Divergence[] = [];
  for (const tool of Object.keys(python)) {
    if (!(tool in typescript)) {
      found.push({ tool, param: null, kind: "missing-ts-tool", observed: "" });
    }
  }
  for (const tool of Object.keys(typescript)) {
    if (!(tool in python)) {
      found.push({ tool, param: null, kind: "missing-py-tool", observed: "" });
    }
  }
  for (const [tool, contract] of Object.entries(python)) {
    const other = typescript[tool];
    if (other) found.push(...compareTool(tool, contract, other, rules));
  }
  return found.sort((a, b) => {
    const [x, y] = [divergenceKey(a), divergenceKey(b)];
    return x < y ? -1 : x > y ? 1 : 0;
  });
}

/** Identity of a divergence, for matching against the accepted-divergence registry. */
export function divergenceKey(d: { tool: string; param: string | null; kind: string }): string {
  return `${d.tool} :: ${d.param ?? ""} :: ${d.kind}`;
}

export function formatDivergence(d: Divergence): string {
  const where = d.param ? `${d.tool}.${d.param}` : d.tool;
  return d.observed ? `${where} [${d.kind}] ${d.observed}` : `${where} [${d.kind}]`;
}
