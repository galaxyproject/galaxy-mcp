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
 *
 * It is a summary of JSON Schema, not an implementation of it, so it fails
 * closed: a construct it has not been taught stops the run or rides along as
 * structure, and it never reports agreement it did not check. It follows no
 * references: a schema is read exactly as the surface wrote it.
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
  $ref?: string;
  $defs?: Record<string, JsonSchema>;
  /** A schema may carry any keyword; what the comparison does with one is its own decision. */
  [keyword: string]: unknown;
}

/** MCP annotations as a tool advertises them. Only the read-only hint is compared. */
export interface ToolAnnotations {
  readOnlyHint?: boolean;
  [hint: string]: unknown;
}

/** One tool as its own surface advertises it. */
export interface ToolContract {
  inputSchema: JsonSchema;
  /** What the tool advertises, not a conclusion already drawn from it. */
  annotations: ToolAnnotations;
  /** Tags, for a surface that advertises them -- FastMCP puts them in `_meta`. */
  tags?: readonly string[];
}

/**
 * Everything one surface advertises, by tool name. A map rather than an object
 * because the names are the surface's own text, and an object answers to names
 * nobody advertised: a tool called `toString` is one an object already has, so on
 * an object it would never read as a tool only one side has.
 */
export type Surface = ReadonlyMap<string, ToolContract>;

export interface Mutability {
  mutating: boolean;
  /** Which of the tool's own statements this reading came from. */
  source: "hint" | "tag" | "mcp default";
}

/**
 * What a tool tells a client about changing things, from the strongest signal it
 * actually advertises: its MCP read-only hint, else a `write` tag, else MCP's own
 * default, which is that a tool without a `readOnlyHint` may change anything.
 *
 * Hints beyond `readOnlyHint` are recorded on both sides but not compared; the
 * Python snapshot is what catches those moving.
 */
export function mutability(contract: ToolContract, where: string): Mutability {
  const node = contract as unknown as Record<string, unknown>;
  const annotations = (read(node, "annotations", "object", where) ?? {}) as Record<string, unknown>;
  const hint = read(annotations, "readOnlyHint", "boolean", where);
  if (typeof hint === "boolean") return { mutating: !hint, source: "hint" };
  const tags = read(node, "tags", "names", where) as string[] | undefined;
  if (tags) return { mutating: tags.includes("write"), source: "tag" };
  return { mutating: true, source: "mcp default" };
}

const showMutability = (m: Mutability): string =>
  `${m.mutating ? "write" : "read"} (${m.source})`;

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
  // Shapes are checked before this runs, so anyOf is a list of schemas or absent.
  return Array.isArray(schema.anyOf) && schema.anyOf.some((v) => v.type === "null");
}

/**
 * `T | None` read as plain `T`. A lone surviving branch stands in for the whole
 * union, but only when nothing said beside it is lost: a keyword the union and
 * the branch both set would have to pick a winner, and there is no honest winner,
 * so the union stays and the two shapes are compared as they are written.
 */
function withoutNull(schema: JsonSchema): JsonSchema {
  if (!schema.anyOf) return schema;
  const variants = schema.anyOf.filter((v) => v.type !== "null");
  const beside: JsonSchema = { ...schema };
  delete beside.anyOf;
  const [only] = variants;
  if (!only || variants.length !== 1) return { ...beside, anyOf: variants };
  const clashes = Object.keys(only).some((keyword) => Object.hasOwn(beside, keyword));
  return clashes ? { ...beside, anyOf: variants } : { ...beside, ...only };
}

const quoted = (keyword: string): string => JSON.stringify(keyword);

/**
 * Keywords whose value is another schema. A `properties` entry is a parameter
 * NAME rather than a keyword, which is why it is spelled out here: a parameter
 * called `default` is still a parameter, and one called `$ref` is still a name.
 */
function childSchemas(schema: Record<string, unknown>, where: string): [string, unknown][] {
  const children: [string, unknown][] = [];
  for (const [keyword, value] of Object.entries(schema)) {
    if (keyword === "properties") {
      // A `properties` that is not an object yields no children; reading it as the
      // parameter list is what refuses it, and that happens with a better message.
      if (!isPlainObject(value)) continue;
      for (const [name, each] of Object.entries(value)) {
        children.push([`${where}.properties.${name}`, each]);
      }
    } else if (keyword === "anyOf" || keyword === "items" || keyword === "propertyNames") {
      // A list where one schema was expected is refused elsewhere, but both walkers
      // still have to see inside it, or one of them reads a shape the other cannot.
      if (Array.isArray(value)) {
        value.forEach((each, i) => children.push([`${where}.${keyword}[${i}]`, each]));
      } else {
        children.push([`${where}.${keyword}`, value]);
      }
    } else if (keyword === "additionalProperties" && value && typeof value === "object") {
      children.push([`${where}.additionalProperties`, value]);
    }
  }
  return children;
}

/**
 * A name the comparison can write down. Everything it records -- a registry key, a
 * missing tool's parameter list -- is delimited text, so a name carrying the
 * delimiters could describe a surface that does not exist. Both surfaces name
 * things after identifiers, so this costs nothing and closes that.
 */
const NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;

function assertIsAName(name: string, what: string, where: string): void {
  if (NAME.test(name)) return;
  throw new Error(
    `${where}: ${quoted(name)} is not a name the comparison can write down, and it records ` +
      `every ${what} in text it has to be able to read back`,
  );
}

/** What `type` may say. A token is read back, so the words in it have to be known. */
const TYPE_NAMES = new Set([
  "array",
  "boolean",
  "integer",
  "null",
  "number",
  "object",
  "string",
]);

/**
 * The largest integer a JSON number survives as itself. Past it two different
 * numbers read back the same, so a surface may not carry one -- the Python
 * generator refuses to write one, and this says the same of the ops.
 */
function assertNumbersAreExact(value: unknown, where: string): void {
  if (Array.isArray(value)) {
    value.forEach((each, i) => assertNumbersAreExact(each, `${where}[${i}]`));
  } else if (value && typeof value === "object") {
    for (const [key, each] of Object.entries(value)) assertNumbersAreExact(each, `${where}.${key}`);
  } else if (typeof value === "number" && Number.isInteger(value) && !Number.isSafeInteger(value)) {
    throw new Error(
      `${where} is ${value}, which no JSON reader can hand back unchanged, so the two surfaces ` +
        "cannot be compared on it",
    );
  }
}

export const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

const SHAPES = {
  object: isPlainObject,
  list: (value: unknown) => Array.isArray(value),
  // A list of names is read as a set -- what is required, how a tool is tagged --
  // so a name written twice says something the comparison would never see again.
  names: (value: unknown) =>
    Array.isArray(value) &&
    value.every((v) => typeof v === "string") &&
    new Set(value).size === value.length,
  string: (value: unknown) => typeof value === "string",
  boolean: (value: unknown) => typeof value === "boolean",
} as const;

export type Shape = keyof typeof SHAPES;

const SHAPE_NAMES: Record<Shape, string> = {
  object: "an object",
  list: "a list",
  names: "a list of names, none of them written twice",
  string: "a string",
  boolean: "a boolean",
};

export const describe = (value: unknown): string =>
  value === null
    ? "null"
    : Array.isArray(value)
      ? "a list"
      : typeof value === "object"
        ? "an object"
        : typeof value;

/**
 * A value that may be left out but must not be something else when it is there.
 *
 * Absent is the key not being there. A key written as null, or as a number where
 * a list belongs, is not silence -- the surface said something, and something
 * wrong, so the run stops rather than defaulting it and comparing a tool nobody
 * described. Every place that would otherwise write `?? something` reads through
 * here.
 */
export function read(
  node: Record<string, unknown>,
  key: string,
  shape: Shape,
  where: string,
): unknown {
  // Only what the surface wrote itself: every object already answers to `toString`
  // and `constructor`, and neither is something a surface said.
  if (!Object.hasOwn(node, key) || node[key] === undefined) return undefined;
  const value = node[key];
  if (SHAPES[shape](value)) return value;
  throw new Error(
    `${where}: ${quoted(key)} is ${describe(value)} where the comparison reads ` +
      `${SHAPE_NAMES[shape]}, so it cannot read the tool as written`,
  );
}

/**
 * What a schema's keywords have to be when they are there at all. A map rather
 * than an object because a surface chooses the keyword it looks up: on an object
 * `constructor` would find one that was never put here.
 */
const KEYWORD_SHAPES = new Map<string, Shape>([
  ["anyOf", "list"],
  ["enum", "list"],
  ["items", "object"],
  ["properties", "object"],
  ["propertyNames", "object"],
  ["required", "names"],
  ["type", "string"],
]);

/** References. The comparison reads a schema as written and follows nothing. */
const REFERENCE = new Set(["$defs", "$ref"]);

/**
 * Constructs that change what a value has to satisfy and that the comparison does
 * not read. Neither surface writes one, and a check that quietly summarized them
 * away would be worse than one that stops.
 */
const REFUSED = new Set([
  "allOf",
  "contains",
  "dependentSchemas",
  "else",
  "if",
  "not",
  "oneOf",
  "patternProperties",
  "prefixItems",
  "then",
  "unevaluatedItems",
  "unevaluatedProperties",
]);

/**
 * What a parameter may be made of. `depth` is how far below the parameter itself
 * we are, because a default means something written on the parameter and nothing
 * the comparison can use written anywhere else.
 */
function assertComparable(node: unknown, where: string, depth: number): void {
  if (!isPlainObject(node)) {
    throw new Error(
      `${where}: the comparison reads a schema here, and found ${describe(node)}`,
    );
  }
  const schema = node;
  if (Array.isArray(schema.items)) {
    throw new Error(
      `${where}: ${quoted("items")} is a list of schemas here, and the comparison reads one ` +
        "element schema; teach it that construct before a surface uses one",
    );
  }
  for (const keyword of Object.keys(schema)) {
    const shape = KEYWORD_SHAPES.get(keyword);
    if (shape) read(schema, keyword, shape, where);
    if (keyword === "additionalProperties") {
      const extras = schema[keyword];
      if (typeof extras !== "boolean" && !isPlainObject(extras)) {
        throw new Error(
          `${where}: ${quoted(keyword)} is ${describe(extras)} where the comparison reads a ` +
            "boolean or a schema, so it cannot read the tool as written",
        );
      }
    }
    if (REFERENCE.has(keyword)) {
      throw new Error(
        `${where}: the comparison does not follow references, and ${quoted(keyword)} is one; ` +
          "teach it before a surface uses one",
      );
    }
    if (REFUSED.has(keyword)) {
      throw new Error(
        `${where}: the comparison does not read ${quoted(keyword)}; teach it that construct ` +
          "before a surface uses one",
      );
    }
    if (keyword === "type" && !TYPE_NAMES.has(schema[keyword] as string)) {
      throw new Error(
        `${where}: ${quoted("type")} says ${JSON.stringify(schema[keyword])}, which is not one ` +
          "of the JSON Schema types; teach the comparison that construct before a surface uses one",
      );
    }
    if (keyword === "default" && depth > 0) {
      throw new Error(
        `${where}: this default is not on the parameter itself, and the comparison only reads ` +
          "the one that is",
      );
    }
  }
  for (const [at, child] of childSchemas(schema, where)) assertComparable(child, at, depth + 1);
}

/**
 * The keys the comparison knows how to read at the top of a tool's input schema.
 * Both surfaces use five of them -- FastMCP writes type, properties, required and
 * additionalProperties, zod writes $schema in place of the last -- and the rest is
 * prose that cannot change what the tool takes. Requiredness and the parameter
 * list are read straight off these keys, so a root keyword that rewrites either
 * (allOf, if/then, oneOf) would be read as absent, which is why an unknown one
 * stops the run.
 *
 * Their values are checked to the extent the comparison reads them. `type`,
 * `properties` and `required` say what the tool takes, so each has to be what it
 * claims to be. `additionalProperties` is never compared -- whether a tool
 * accepts parameters it does not declare is outside what this check says -- so a
 * wrong-typed one cannot hide anything in scope, though when it is a schema it is
 * walked like any other. `$schema`, `title` and `description` are never read at
 * all.
 */
const ROOT_KEYS = new Set([
  "$schema",
  "additionalProperties",
  "description",
  "properties",
  "required",
  "title",
  "type",
]);

/** A tool's input schema, once it is something the comparison can honestly read. */
export function readInputSchema(schema: JsonSchema, where: string): JsonSchema {
  const stray = Object.keys(schema).filter((keyword) => !ROOT_KEYS.has(keyword));
  const references = stray.filter((keyword) => REFERENCE.has(keyword));
  if (references.length) {
    throw new Error(
      `${where}: ${references.map(quoted).join(", ")} at the top of an input schema -- the ` +
        "comparison does not follow references; teach it before a surface uses one",
    );
  }
  if (stray.length) {
    throw new Error(
      `${where}: the comparison does not model ${stray.map(quoted).join(", ")} at the top of an ` +
        "input schema, so it cannot say what the tool takes; teach it that keyword before the " +
        "surface uses one",
    );
  }
  // MCP tool inputs are objects, and everything below reads the root as one: a root
  // that says otherwise describes a tool nobody can call the way the check assumes.
  if (schema.type !== "object") {
    throw new Error(
      `${where}: the root says ${JSON.stringify(schema.type)}, and the comparison only reads a ` +
        "tool that takes an object",
    );
  }
  const declaredParameters = read(schema, "properties", "object", where) ?? {};
  const names = (read(schema, "required", "names", where) ?? []) as string[];
  // Everything below a parameter is walked parameter by parameter, under its own
  // name; this covers what the root says about anything else, such as the shape of
  // the properties it does not declare.
  const beyondTheParameters: Record<string, unknown> = { ...schema };
  delete beyondTheParameters.properties;
  assertComparable(beyondTheParameters, where, 0);
  // Requiredness is read off the parameters, so a required name that is not one
  // of them is a demand the comparison would never see.
  const declared = Object.keys(declaredParameters);
  const orphans = names.filter((name) => !declared.includes(name));
  if (orphans.length) {
    throw new Error(
      `${where}: ${orphans.map(quoted).join(", ")} is required but is not one of the parameters ` +
        "this schema declares, and the comparison has no way to say what the tool takes",
    );
  }
  return schema;
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
  // JSON.stringify gives undefined back for undefined, which is not the same thing
  // as a written null and must not render as one.
  return JSON.stringify(value) ?? "undefined";
}

/** The keywords the token speaks for. */
const MODELED = new Set(["anyOf", "enum", "items", "type"]);

/**
 * The keywords the comparison drops on purpose: prose, value constraints, and
 * anything about the inside of an object -- an open record and a closed one both
 * read `object`, and Python spells its open records `additionalProperties: true`
 * where zod spells them `propertyNames` plus an empty `additionalProperties`.
 * A keyword in neither list is rendered beside the token as structure rather than
 * assumed to be harmless, and a test asks both surfaces not to use one until
 * somebody has said which list it belongs in.
 */
const IGNORED = new Set([
  "$schema",
  "additionalProperties",
  "default",
  "description",
  "examples",
  "exclusiveMaximum",
  "exclusiveMinimum",
  "format",
  "maxItems",
  "maxLength",
  "maximum",
  "minItems",
  "minLength",
  "minimum",
  "multipleOf",
  "pattern",
  "properties",
  "propertyNames",
  "required",
  "title",
  "uniqueItems",
]);

/**
 * A comparable shorthand for a parameter's type, e.g. `string`, `array&items<string>`.
 *
 * It models the constructs both surfaces actually use. It does NOT compare
 * value constraints (format, minimum, pattern), `additionalProperties`, or the
 * inner shape of an object -- an open record and a closed one both read `object`.
 * Anything else it does not model rides along as structure, so a restriction it
 * has never heard of cannot make two unlike shapes match by accident. A schema
 * with nothing modeled in it at all is compared whole, ignored keywords and all,
 * which can over-report but cannot hide anything.
 */
export function typeToken(schema: JsonSchema): string {
  const parts: string[] = [];
  // What the token spoke for, rather than what it might have: a keyword it knows
  // but could not read -- `type: ""` -- has to ride along like any other.
  const spoken = new Set<string>();
  const speak = (keyword: string, part: string) => {
    parts.push(part);
    spoken.add(keyword);
  };
  if (schema.enum) {
    speak("enum", `enum<${schema.enum.map((v) => JSON.stringify(v)).sort().join("|")}>`);
  }
  if (schema.anyOf) {
    speak("anyOf", `anyOf<${schema.anyOf.map(typeToken).sort().join("|")}>`);
  }
  // Every keyword reads the same wherever it sits: an element type beside a union
  // says as much about what the parameter takes as one beside `type: "array"`.
  if (schema.items) speak("items", `items<${typeToken(schema.items)}>`);
  if (schema.type) speak("type", schema.type);
  const carried = Object.entries(schema).filter(
    ([keyword]) => !spoken.has(keyword) && !IGNORED.has(keyword),
  );
  if (carried.length) parts.push(`also(${stableStringify(Object.fromEntries(carried))})`);
  if (parts.length) return parts.sort().join("&");
  return `schema(${stableStringify(schema)})`;
}

export interface UnknownKeyword {
  /** The path to the schema that used it, for a message that can be acted on. */
  where: string;
  keyword: string;
}

/**
 * Every keyword in a schema that is neither modeled nor deliberately ignored.
 *
 * One of those still rides along as structure, so it cannot hide a difference
 * between the surfaces; what it cannot do is mean anything. A surface that starts
 * emitting one -- a zod or FastMCP upgrade usually -- should say so by name
 * rather than by being compared as though the keyword were not there.
 */
export function unknownKeywords(schema: unknown, where: string): UnknownKeyword[] {
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) return [];
  const node = schema as Record<string, unknown>;
  const found: UnknownKeyword[] = [];
  for (const keyword of Object.keys(node)) {
    if (!MODELED.has(keyword) && !IGNORED.has(keyword)) found.push({ where, keyword });
  }
  for (const [at, child] of childSchemas(node, where)) found.push(...unknownKeywords(child, at));
  return found;
}

export function normalizeParams(
  schema: JsonSchema,
  rules: Normalization,
  where: string,
): Map<string, NormalParam> {
  const root = readInputSchema(schema, where) as unknown as Record<string, unknown>;
  const required = new Set((read(root, "required", "names", where) ?? []) as string[]);
  const out = new Map<string, NormalParam>();
  const declared = (read(root, "properties", "object", where) ?? {}) as Record<string, JsonSchema>;
  for (const [rawName, prop] of Object.entries(declared)) {
    assertIsAName(rawName, "parameter", where);
    assertComparable(prop, `${where} ${quoted(rawName)}`, 0);
    assertNumbersAreExact(prop, `${where} ${quoted(rawName)}`);
    const isRequired = required.has(rawName);
    const optionalNull = !isRequired && isNullable(prop);
    const typeSchema = rules.optionalNullUnions && optionalNull ? withoutNull(prop) : prop;
    const hasDefault =
      Object.hasOwn(prop, "default") &&
      !(rules.pythonNullDefaults && optionalNull && prop.default === null);
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

const showContract = (p: NormalParam): string =>
  `type=${p.type} required=${p.required} default=${show(p)}`;

/**
 * Everything the one surface that has a tool says about it. A tool the other
 * surface lacks has no comparison to go stale, so what the side that does have
 * it advertises is the entry's only anchor: change the tool and the registry
 * entry stops describing it.
 */
function showTool(contract: ToolContract, rules: Normalization, where: string): string {
  const params = [...normalizeParams(contract.inputSchema, rules, where)]
    .sort(([a], [b]) => (a < b ? -1 : 1))
    .map(([name, p]) => `${name} ${showContract(p)}`)
    .join("; ");
  return `${showMutability(mutability(contract, where))} params=[${params}]`;
}

function compareTool(
  tool: string,
  python: ToolContract,
  typescript: ToolContract,
  rules: Normalization,
): Divergence[] {
  const found: Divergence[] = [];
  const [pySays, tsSays] = [
    mutability(python, `${tool} (python)`),
    mutability(typescript, `${tool} (typescript)`),
  ];
  if (pySays.mutating !== tsSays.mutating) {
    found.push({
      tool,
      param: null,
      kind: "mutability-mismatch",
      observed: `python=${showMutability(pySays)} typescript=${showMutability(tsSays)}`,
    });
  }
  const py = normalizeParams(python.inputSchema, rules, `${tool} (python)`);
  const ts = normalizeParams(typescript.inputSchema, rules, `${tool} (typescript)`);
  for (const [param, p] of py) {
    if (!ts.has(param)) {
      found.push({
        tool,
        param,
        kind: "missing-ts-param",
        observed: `python=${showContract(p)}`,
      });
    }
  }
  for (const [param, t] of ts) {
    if (!py.has(param)) {
      found.push({
        tool,
        param,
        kind: "missing-py-param",
        observed: `typescript=${showContract(t)}`,
      });
    }
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
  python: Surface,
  typescript: Surface,
  rules: Normalization,
): Divergence[] {
  const found: Divergence[] = [];
  for (const tool of [...python.keys(), ...typescript.keys()]) {
    assertIsAName(tool, "tool", "the surfaces");
  }
  for (const [tool, contract] of python) {
    if (typescript.has(tool)) continue;
    const observed = `python=${showTool(contract, rules, `${tool} (python)`)}`;
    found.push({ tool, param: null, kind: "missing-ts-tool", observed });
  }
  for (const [tool, contract] of typescript) {
    if (python.has(tool)) continue;
    const observed = `typescript=${showTool(contract, rules, `${tool} (typescript)`)}`;
    found.push({ tool, param: null, kind: "missing-py-tool", observed });
  }
  for (const [tool, contract] of python) {
    const other = typescript.get(tool);
    if (other) found.push(...compareTool(tool, contract, other, rules));
  }
  return found.sort((a, b) => {
    const [x, y] = [divergenceKey(a), divergenceKey(b)];
    return x < y ? -1 : x > y ? 1 : 0;
  });
}

/**
 * A surface keyed by tool name. Two entries of one name would fold into one and
 * the other would never be compared, so the run stops instead of choosing.
 */
export function surfaceByName(entries: [string, ToolContract][], where: string): Surface {
  const twice = entries.map(([name]) => name).filter((name, i, all) => all.indexOf(name) !== i);
  if (twice.length) {
    throw new Error(
      `${where}: ${[...new Set(twice)].map(quoted).join(", ")} is listed more than once, and the ` +
        "comparison has no way to say which of them a client is given",
    );
  }
  return new Map(entries);
}

/** Identity of a divergence, for matching against the accepted-divergence registry. */
export function divergenceKey(d: { tool: string; param: string | null; kind: string }): string {
  return `${d.tool} :: ${d.param ?? ""} :: ${d.kind}`;
}

export function formatDivergence(d: Divergence): string {
  const where = d.param ? `${d.tool}.${d.param}` : d.tool;
  return d.observed ? `${where} [${d.kind}] ${d.observed}` : `${where} [${d.kind}]`;
}
