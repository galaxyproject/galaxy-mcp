/**
 * Contract comparison between the Python MCP server's generated surface manifest
 * (mcp-server-galaxy-py/tests/testdata/mcp-surface.json) and the tools this
 * package advertises.
 *
 * Both sides describe themselves as JSON Schema, so the comparison is over
 * parameter names, types, requiredness and declared defaults. Descriptions are
 * deliberately not compared: the two surfaces word things differently on purpose
 * and diffing prose would bury the contract differences that matter.
 */

export type DivergenceKind =
  | "missing-ts-tool"
  | "missing-py-tool"
  | "missing-ts-param"
  | "missing-py-param"
  | "type-mismatch"
  | "required-mismatch"
  | "default-mismatch";

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
  properties?: Record<string, JsonSchema>;
  required?: string[];
  default?: unknown;
}

/**
 * Whole-surface differences that would otherwise produce a divergence per
 * parameter. Each is a single switch, flipped from the registry.
 */
export interface Normalization {
  /** Compare TS `historyId` with Python `history_id` as the same parameter. */
  snakeCaseParamNames: boolean;
  /** Read Python's `x: str | None = None` as plain optional rather than "defaults to null". */
  pythonNullDefaults: boolean;
}

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

/** A comparable shorthand for a parameter's type, e.g. `string`, `array<string>`. */
export function typeToken(schema: JsonSchema): string {
  const s = withoutNull(schema);
  if (s.anyOf) {
    return `anyOf<${s.anyOf.map(typeToken).sort().join("|")}>`;
  }
  if (s.type === "array") return `array<${typeToken(s.items ?? {})}>`;
  return s.type ?? "any";
}

export function normalizeParams(schema: JsonSchema, rules: Normalization): Map<string, NormalParam> {
  const required = new Set(schema.required ?? []);
  const out = new Map<string, NormalParam>();
  for (const [rawName, prop] of Object.entries(schema.properties ?? {})) {
    const isRequired = required.has(rawName);
    const nullDefault = prop.default === null && isNullable(prop);
    const hasDefault =
      "default" in prop && !(rules.pythonNullDefaults && !isRequired && nullDefault);
    out.set(rules.snakeCaseParamNames ? toSnakeCase(rawName) : rawName, {
      type: typeToken(prop),
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
  python: JsonSchema,
  typescript: JsonSchema,
  rules: Normalization,
): Divergence[] {
  const py = normalizeParams(python, rules);
  const ts = normalizeParams(typescript, rules);
  const found: Divergence[] = [];
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
  python: Record<string, JsonSchema>,
  typescript: Record<string, JsonSchema>,
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
  for (const [tool, schema] of Object.entries(python)) {
    const other = typescript[tool];
    if (other) found.push(...compareTool(tool, schema, other, rules));
  }
  return found.sort((a, b) => divergenceKey(a).localeCompare(divergenceKey(b)));
}

/** Identity of a divergence, for matching against the accepted-divergence registry. */
export function divergenceKey(d: { tool: string; param: string | null; kind: string }): string {
  return `${d.tool} :: ${d.param ?? ""} :: ${d.kind}`;
}

export function describe(d: Divergence): string {
  const where = d.param ? `${d.tool}.${d.param}` : d.tool;
  return d.observed ? `${where} [${d.kind}] ${d.observed}` : `${where} [${d.kind}]`;
}
