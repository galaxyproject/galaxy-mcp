import { beforeAll, describe, expect, it } from "vitest";
import { toolNames } from "../src/server";
import {
  compareSurfaces,
  divergenceKey,
  formatDivergence,
  NORMALIZATION_RULES,
  typeToken,
  unknownKeywords,
  DIVERGENCE_KINDS,
  EITHER_WAY_KINDS,
  WHOLE_TOOL_KINDS,
  type Divergence,
  type JsonSchema,
  type Normalization,
  type Surface,
  type ToolContract,
} from "./parity/compare";
import {
  DIVERGENCE_STATUSES,
  loadManifest,
  loadRegistry,
  normalizationFrom,
  pythonSurface,
  ratchetProblems,
  RATCHETED_STATUS,
  typescriptSurface,
  type AcceptedDivergence,
  type Manifest,
  type Registry,
} from "./parity/surfaces";



/** The switches the check itself runs with, so a test cannot prove a shape CI never compares. */
const RULES: Normalization = normalizationFrom(loadRegistry());

/**
 * The comparison, over two surfaces written here as object literals: their own
 * entries, which is what `surfaceByName` builds out of what a surface advertises.
 * A tool named `__proto__` cannot come through here -- written in a literal it
 * sets the prototype instead of declaring anything -- so the tests about names
 * like that hand `compareSurfaces` a map themselves.
 */
const compare = (
  python: Record<string, ToolContract>,
  typescript: Record<string, ToolContract>,
  rules: Normalization,
): Divergence[] =>
  compareSurfaces(new Map(Object.entries(python)), new Map(Object.entries(typescript)), rules);

const where = (d: { tool: string; param: string | null; kind: string }) =>
  `${d.tool}${d.param ? `.${d.param}` : ""} [${d.kind}]`;

let manifest: Manifest;
let registry: Registry;
let advertised: Surface;
let found: Divergence[];

beforeAll(async () => {
  manifest = loadManifest();
  registry = loadRegistry();
  advertised = await typescriptSurface();
  found = compareSurfaces(pythonSurface(manifest), advertised, normalizationFrom(registry));
});

describe("contract parity with the Python MCP server", () => {
  it("has no divergence that is missing from the accepted-divergence registry", () => {
    const accepted = new Set(registry.divergences.map(divergenceKey));
    const unregistered = found.filter((d) => !accepted.has(divergenceKey(d)));
    expect(
      unregistered.map(formatDivergence),
      `${unregistered.length} unregistered divergence(s) between the TS ops and the Python ` +
        "server. Review each one and add it to test/fixtures/accepted-divergences.json, or " +
        "close the gap in the op.",
    ).toEqual([]);
  });

  it("has no registry entry the two surfaces no longer disagree about", () => {
    const current = new Set(found.map(divergenceKey));
    const stale = registry.divergences.filter((d) => !current.has(divergenceKey(d)));
    expect(
      stale.map(where),
      "these registry entries no longer describe a real divergence -- delete them",
    ).toEqual([]);
  });

  it("has registry entries that still describe what each side says", () => {
    const byKey = new Map(found.map((d) => [divergenceKey(d), d]));
    const moved = registry.divergences
      .map((entry) => ({ entry, actual: byKey.get(divergenceKey(entry)) }))
      .filter(({ entry, actual }) => actual && actual.observed !== entry.observed)
      .map(
        ({ entry, actual }) =>
          `${where(entry)}: registry says "${entry.observed}", surfaces say "${actual?.observed}"`,
      );
    expect(
      moved,
      "one side of these divergences changed; re-review the entry and update `observed`",
    ).toEqual([]);
  });
});

describe("the accepted-divergence registry itself", () => {
  it("is well formed", () => {
    const problems: string[] = [];
    const seen = new Set<string>();
    for (const entry of registry.divergences) {
      if (!DIVERGENCE_KINDS.includes(entry.kind)) problems.push(`${where(entry)}: unknown kind`);
      if (!DIVERGENCE_STATUSES.includes(entry.status)) {
        problems.push(`${where(entry)}: unknown status`);
      }
      if (!entry.reason?.trim()) problems.push(`${where(entry)}: needs a reason`);
      if (typeof entry.observed !== "string") {
        problems.push(`${where(entry)}: needs an observed string`);
      }
      if (
        !EITHER_WAY_KINDS.includes(entry.kind) &&
        WHOLE_TOOL_KINDS.includes(entry.kind) !== (entry.param === null)
      ) {
        problems.push(`${where(entry)}: param must be null for whole-tool kinds and set otherwise`);
      }
      const key = divergenceKey(entry);
      if (seen.has(key)) problems.push(`${where(entry)}: duplicate entry`);
      seen.add(key);
    }
    expect(problems).toEqual([]);
  });

  it("declares every normalization rule the comparator applies", () => {
    // Keyed by names the registry chose, so it is read as a map like the rest.
    const written = new Map(Object.entries(registry.normalization));
    expect(
      [...written.keys()].sort(),
      "the registry must declare exactly the rules the comparator knows about",
    ).toEqual([...NORMALIZATION_RULES].sort());
    for (const name of NORMALIZATION_RULES) {
      const rule = written.get(name);
      expect(rule, `${name} is not declared`).toBeDefined();
      expect(typeof rule?.enabled, `${name}.enabled`).toBe("boolean");
      expect(DIVERGENCE_STATUSES, `${name}.status`).toContain(rule?.status);
      expect(rule?.reason.trim().length, `${name}.reason`).toBeGreaterThan(0);
    }
  });

  it("covers every kind the comparator can report", () => {
    expect([...DIVERGENCE_KINDS].sort()).toEqual([...new Set(DIVERGENCE_KINDS)].sort());
    // Both lists name kinds; a kind in neither would be reported and never checked for shape.
    for (const kind of [...WHOLE_TOOL_KINDS, ...EITHER_WAY_KINDS]) {
      expect(DIVERGENCE_KINDS).toContain(kind);
    }
  });
});

describe("the ratchet on unreviewed gaps", () => {
  const gap: AcceptedDivergence = {
    tool: "example",
    param: "limit",
    kind: "missing-ts-param",
    observed: "python=type=integer required=false default=none",
    status: RATCHETED_STATUS,
    reason: "the comparator found it and nobody has read both sides yet",
  };
  const withEntries = (...extra: AcceptedDivergence[]): Registry => ({
    ...registry,
    divergences: [...registry.divergences, ...extra],
  });

  /**
   * A registry made up for the occasion: so many gaps, so many allowed. The live one
   * says what parity is today and will one day legitimately hold none, which is no
   * basis for a test about what the ratchet does when the two numbers disagree.
   */
  const holding = (gaps: number, allowed: number): Registry => ({
    ...registry,
    ratchet: { unreviewedGaps: allowed },
    divergences: Array.from({ length: gaps }, (_, index) => ({
      ...gap,
      param: `limit_${index}`,
    })),
  });

  it("holds the registry at the number of unreviewed gaps it declares", () => {
    expect(
      ratchetProblems(registry),
      "the registry and the number it pins itself at have come apart",
    ).toEqual([]);
  });

  it("is tripped by one more unreviewed gap", () => {
    expect(ratchetProblems(holding(4, 3))).toEqual([
      expect.stringContaining("is the most it may hold"),
    ]);
    // And on the registry as it stands, whatever number that is today.
    expect(ratchetProblems(withEntries(gap))).toEqual([
      expect.stringContaining("is the most it may hold"),
    ]);
  });

  it("lets a reviewed status through, however many there are", () => {
    expect(ratchetProblems(withEntries({ ...gap, status: "pending-port" }))).toEqual([]);
    expect(ratchetProblems(withEntries({ ...gap, status: "pending-decision" }))).toEqual([]);
    expect(ratchetProblems(withEntries({ ...gap, status: "intentional" }))).toEqual([]);
  });

  it("asks for the number to come down when a gap is closed", () => {
    // A made-up registry, not the live one: closing the last real gap is a good day,
    // not a build failure, and the live number will legitimately reach zero.
    expect(ratchetProblems(holding(2, 3))).toEqual([expect.stringContaining("Lower")]);
    expect(ratchetProblems(holding(0, 1))).toEqual([expect.stringContaining("Lower")]);
    expect(ratchetProblems(holding(0, 0))).toEqual([]);
  });

  it("stops the run when the registry declares no number at all", () => {
    const { ratchet: _none, ...silent } = registry;
    expect(() => ratchetProblems(silent as Registry)).toThrow(/how many.*may hold/s);
  });

  it("stops the run when the number is not a count", () => {
    const wrong = { ...registry, ratchet: { unreviewedGaps: "24" } } as unknown as Registry;
    expect(() => ratchetProblems(wrong)).toThrow(/"unreviewedGaps" is string.*a count/s);
    // A number that is not a number of things: each would otherwise compare against the
    // real count and quietly pass or quietly fail on an arithmetic nobody meant.
    for (const nonsense of [-1, 24.5, Number.NaN, 2 ** 53, Number.POSITIVE_INFINITY]) {
      const declared = { ...registry, ratchet: { unreviewedGaps: nonsense } };
      expect(() => ratchetProblems(declared), String(nonsense)).toThrow(/a count/);
    }
  });
});

describe("missing tool divergences", () => {
  it("makes a registered missing tool stale when the side that has it changes", () => {
    const typescript: Record<string, ToolContract> = {};
    const example: ToolContract = {
      inputSchema: {
        type: "object",
        properties: { title: { type: "string" } },
        required: ["title"],
      },
      annotations: {},
      tags: ["read"],
    };
    const python: Record<string, ToolContract> = { example };
    const accepted = "python=read (tag) params=[title type=string required=true default=none]";

    expect(compare(python, typescript, RULES)).toContainEqual(
      expect.objectContaining({ kind: "missing-ts-tool", observed: accepted }),
    );

    example.inputSchema.properties!.title = { type: "integer" };
    example.inputSchema.properties!.slug = { type: "string", default: "x" };
    example.tags = ["write"];

    const changed = compare(python, typescript, RULES);
    expect(changed).toContainEqual(
      expect.objectContaining({
        observed:
          "python=write (tag) params=[slug type=string required=false default=\"x\"; " +
          "title type=integer required=true default=none]",
      }),
    );
    expect(changed.map((d) => d.observed)).not.toContain(accepted);
  });

  it("records the side that has the tool when TypeScript is the one that has it", () => {
    const python: Record<string, ToolContract> = {};
    const typescript: Record<string, ToolContract> = {
      example: {
        inputSchema: { type: "object", properties: { limit: { type: "integer", default: 10 } } },
        annotations: { readOnlyHint: true },
      },
    };

    expect(compare(python, typescript, RULES)).toEqual([
      expect.objectContaining({
        kind: "missing-py-tool",
        observed: "typescript=read (hint) params=[limit type=integer required=false default=10]",
      }),
    ]);
  });
});

describe("missing parameter divergences", () => {
  it("makes a registered missing parameter stale when its existing contract changes", () => {
    const typescript: Record<string, ToolContract> = {
      example: { inputSchema: { type: "object" }, annotations: {} },
    };
    const accepted = "python=type=integer required=false default=10";
    const example: ToolContract = {
      inputSchema: {
        type: "object",
        properties: { preview_lines: { default: 10, type: "integer" } },
      },
      annotations: {},
    };
    const python: Record<string, ToolContract> = { example };

    expect(compare(python, typescript, RULES)).toContainEqual(
      expect.objectContaining({ observed: accepted }),
    );

    example.inputSchema.properties!.preview_lines = {
      default: "all",
      type: "string",
    };
    example.inputSchema.required = ["preview_lines"];

    const changed = compare(python, typescript, RULES);
    expect(changed).toContainEqual(
      expect.objectContaining({
        observed: "python=type=string required=true default=\"all\"",
      }),
    );
    expect(changed.map((d) => d.observed)).not.toContain(accepted);
  });
});

describe("the top of an input schema", () => {
  const composed = (): ToolContract => ({
    inputSchema: {
      type: "object",
      properties: { title: { type: "string" } },
      allOf: [{ required: ["title"] }],
    },
    annotations: {},
    tags: ["write"],
  });

  it("stops the run on a keyword it cannot read, naming the tool and the keyword", () => {
    expect(() => compare({ create_page: composed() }, {}, RULES)).toThrow(
      /create_page \(python\).*"allOf"/s,
    );
  });

  it("stops the run for a tool both surfaces have", () => {
    const shared: Record<string, ToolContract> = { invoke_workflow: composed() };
    expect(() =>
      compare(shared, { invoke_workflow: { inputSchema: {}, annotations: {} } }, RULES),
    ).toThrow(/invoke_workflow \(python\).*"allOf"/s);
  });

  it("reads the keys both surfaces actually put there", () => {
    const python: Record<string, ToolContract> = {
      example: {
        inputSchema: {
          $schema: "https://json-schema.org/draft/2020-12/schema",
          title: "Example",
          description: "prose",
          type: "object",
          additionalProperties: false,
          properties: { title: { type: "string" } },
          required: ["title"],
        },
        annotations: {},
      },
    };

    expect(compare(python, python, RULES)).toEqual([]);
  });
});

describe("shapes the comparison will not guess at", () => {
  it("stops the run on a required name that is not a parameter", () => {
    const python: Record<string, ToolContract> = {
      create_page: {
        inputSchema: {
          type: "object",
          additionalProperties: false,
          properties: { title: { type: "string" } },
          required: ["title", "token"],
        },
        annotations: {},
        tags: ["write"],
      },
    };

    expect(() => compare(python, {}, RULES)).toThrow(
      /create_page \(python\).*"token"/s,
    );
  });

  it("stops the run on a default that is not on the parameter itself", () => {
    const python: Record<string, ToolContract> = {
      example: {
        inputSchema: {
          type: "object",
          properties: { value: { anyOf: [{ type: "string", default: "a" }, { type: "integer" }] } },
          required: ["value"],
        },
        annotations: {},
      },
    };

    expect(() => compare(python, {}, RULES)).toThrow(
      /example \(python\).*"value".*default/s,
    );
  });
});

describe("references, which the comparison does not follow", () => {
  const tool = (inputSchema: Record<string, unknown>): Record<string, ToolContract> => ({
    example: { inputSchema, annotations: {}, tags: ["read"] },
  });

  it("stops the run on a reference wherever it sits", () => {
    expect(() =>
      compare(
        tool({ type: "object", properties: { v: { $ref: "#/$defs/V" } } }),
        {},
        RULES,
      ),
    ).toThrow(/example \(python\).*does not follow references/s);
  });

  it("stops the run on definitions at the top of the schema", () => {
    expect(() =>
      compare(
        tool({ type: "object", $defs: { V: { type: "string" } }, properties: {} }),
        {},
        RULES,
      ),
    ).toThrow(/example \(python\).*\$defs/s);
  });

  it("reads a parameter's name as a name, not as a keyword", () => {
    // A parameter called `default` is a parameter; its schema still has to be read.
    const named = (type: string) =>
      tool({
        type: "object",
        properties: { default: { $ref: "#/$defs/Value" }, other: { type } },
      });
    expect(() => compare(named("string"), {}, RULES)).toThrow(
      /example \(python\).*does not follow references/s,
    );
  });

  it("does not mistake a parameter named like a keyword for one", () => {
    const python = tool({
      type: "object",
      properties: { default: { type: "integer" }, type: { type: "string" } },
      required: ["type"],
    });

    expect(compare(python, {}, RULES)).toEqual([
      expect.objectContaining({
        kind: "missing-ts-tool",
        observed:
          "python=read (tag) params=[default type=integer required=false default=none; " +
          "type type=string required=true default=none]",
      }),
    ]);
  });

  it("reads a parameter called $ref as a name, and refuses it as one", () => {
    const python = tool({ type: "object", properties: { $ref: { type: "string" } } });

    expect(() => compare(python, {}, RULES)).toThrow(/is not a name/);
  });

  it("stops the run on a reference that points through another one", () => {
    const chained = tool({
      type: "object",
      $defs: { A: { $ref: "#/$defs/B" }, B: { type: "integer" } },
      properties: { title: { $ref: "#/$defs/A", type: "string" } },
    });

    expect(() => compare(chained, {}, RULES)).toThrow(/example \(python\)/);
  });

  it("stops the run on a pointer into a definition", () => {
    const deep = tool({
      type: "object",
      $defs: { Foo: { items: { type: "integer" } }, "Foo/items": { type: "string" } },
      properties: { title: { $ref: "#/$defs/Foo/items" } },
    });

    expect(() => compare(deep, {}, RULES)).toThrow(/example \(python\)/);
  });
});

describe("the values at the top of an input schema", () => {
  const root = (extra: Record<string, unknown>): Record<string, ToolContract> => ({
    create_page: {
      inputSchema: { type: "object", properties: { title: { type: "string" } }, ...extra },
      annotations: {},
      tags: ["write"],
    },
  });

  it("stops the run when the root is not an object", () => {
    expect(() => compare(root({ type: "array" }), {}, RULES)).toThrow(
      /create_page \(python\).*takes an object/s,
    );
  });

  it("walks what the root says about the properties it does not declare", () => {
    const open = root({ additionalProperties: { type: "array", items: [{ type: "string" }] } });
    expect(() => compare(open, {}, RULES)).toThrow(
      /create_page \(python\).*"items"/s,
    );
  });

  it("stops the run when the required list is not a list of names", () => {
    expect(() => compare(root({ required: "title" }), {}, RULES)).toThrow(
      /create_page \(python\).*"required".*names/s,
    );
  });

  it("stops the run when the parameters are not written as an object", () => {
    const listed = {
      create_page: {
        inputSchema: { type: "object", properties: [] },
        annotations: {},
        tags: ["write"],
      },
    } as unknown as Record<string, ToolContract>;
    expect(() => compare(listed, {}, RULES)).toThrow(
      /create_page \(python\).*"properties"/s,
    );
  });
});

describe("a list that is read as a set", () => {
  it("stops the run on a required name written twice", () => {
    const twice = {
      get_page: {
        inputSchema: {
          type: "object",
          properties: { page_id: { type: "string" } },
          required: ["page_id", "page_id"],
        },
        annotations: {},
        tags: ["read"],
      },
    } as unknown as Record<string, ToolContract>;

    expect(() => compare(twice, {}, RULES)).toThrow(
      /get_page \(python\).*"required".*twice/s,
    );
  });

  it("keeps a repeat inside a list whose repeats are still compared", () => {
    // enum members and union branches keep their multiplicity in the token, so a
    // repeat there is a difference the comparison can still see.
    expect(typeToken({ enum: ["a", "a"] })).not.toBe(typeToken({ enum: ["a"] }));
    expect(typeToken({ anyOf: [{ type: "string" }, { type: "string" }] })).not.toBe(
      typeToken({ anyOf: [{ type: "string" }] }),
    );
  });

  it("stops the run on two manifest entries of one name", () => {
    const tool = (name: string) => ({
      name,
      tags: ["read"],
      annotations: {},
      inputSchema: { type: "object", properties: {} },
    });
    const twice = { source: "x", toolCount: 2, tools: [tool("get_page"), tool("get_page")] };

    expect(() => pythonSurface(twice as unknown as Manifest)).toThrow(/get_page.*more than once/);
  });
});

describe("a value that is there but is not what it claims", () => {
  const say = (inputSchema: Record<string, unknown>, rest: Record<string, unknown> = {}) =>
    ({
      create_page: { inputSchema, annotations: {}, tags: ["write"], ...rest },
    }) as unknown as Record<string, ToolContract>;
  const withTitle = (extra: Record<string, unknown>) => ({
    type: "object",
    properties: { title: { type: "string" } },
    ...extra,
  });

  it("does not read a null required list as no required list", () => {
    expect(() => compare(say(withTitle({ required: null })), {}, RULES)).toThrow(
      /create_page \(python\).*"required" is null/s,
    );
  });

  it("does not read a null union as no union", () => {
    const schema = withTitle({ properties: { title: { type: "string", anyOf: null } } });
    expect(() => compare(say(schema), {}, RULES)).toThrow(/"anyOf" is null/);
  });

  it("stops the run on a parameter that is not a schema", () => {
    const schema = withTitle({ properties: { title: true } });
    expect(() => compare(say(schema), {}, RULES)).toThrow(
      /create_page \(python\).*"title".*boolean/s,
    );
  });

  it("stops the run on annotations that are not annotations", () => {
    expect(() =>
      compare(say(withTitle({}), { annotations: "read-only" }), {}, RULES),
    ).toThrow(/"annotations" is string/);
  });

  it("does not let a hint that is not a hint fall through to the tags", () => {
    expect(() =>
      compare(say(withTitle({}), { annotations: { readOnlyHint: "true" } }), {}, RULES),
    ).toThrow(/"readOnlyHint" is string/);
  });

  it("stops the run on tags that are not a list of tags", () => {
    expect(() => compare(say(withTitle({}), { tags: "write" }), {}, RULES)).toThrow(
      /"tags" is string/,
    );
  });
});

describe("text the surfaces choose themselves", () => {
  const withParams = (properties: Record<string, unknown>, tool = "example") => ({
    [tool]: {
      inputSchema: { type: "object", properties },
      annotations: {},
      tags: ["read"],
    } as ToolContract,
  });

  it("stops the run on a parameter name that is not a name", () => {
    const forged = "annotation type=string required=false default=none; content";
    expect(() =>
      compare(withParams({ [forged]: { type: "string" } }), {}, RULES),
    ).toThrow(/example \(python\).*is not a name the comparison can write down/s);
  });

  it("stops the run on a tool name that is not a name", () => {
    const forged = "create_page :: title :: type-mismatch";
    expect(() => compare(withParams({ a: { type: "string" } }, forged), {}, RULES)).toThrow(
      /is not a name the comparison can write down/,
    );
  });

  it("stops the run on a type that is not one of the JSON Schema types", () => {
    expect(() =>
      compare(
        withParams({ a: { type: "string required=false default=none; b" } }),
        {},
        RULES,
      ),
    ).toThrow(/example \(python\) "a".*"type"/s);
  });

  it("stops the run on a number JSON cannot hand back unchanged", () => {
    expect(() =>
      compare(withParams({ a: { type: "integer", default: 2 ** 53 } }), {}, RULES),
    ).toThrow(/example \(python\) "a".*unchanged/s);
  });

  it("lets the ordinary names through", () => {
    const python = withParams({ history_id: { type: "string" }, _private: { type: "integer" } });
    expect(compare(python, {}, RULES)).toEqual([
      expect.objectContaining({ kind: "missing-ts-tool" }),
    ]);
  });
});

describe("names every JavaScript object already answers to", () => {
  // Legal names both surfaces could pick, and names an empty object reports having.
  const INHERITED = ["toString", "constructor", "hasOwnProperty", "valueOf", "__proto__"];

  const tool = (): ToolContract => ({
    inputSchema: { type: "object", properties: { value: { type: "string" } } },
    annotations: { readOnlyHint: true },
  });
  const params = "params=[value type=string required=false default=none]";

  it("reports a tool only TypeScript advertises, whatever it is called", () => {
    for (const name of INHERITED) {
      expect(compareSurfaces(new Map(), new Map([[name, tool()]]), RULES), name).toEqual([
        {
          tool: name,
          param: null,
          kind: "missing-py-tool",
          observed: `typescript=read (hint) ${params}`,
        },
      ]);
    }
  });

  it("reports a tool only Python has, whatever it is called", () => {
    for (const name of INHERITED) {
      expect(compareSurfaces(new Map([[name, tool()]]), new Map(), RULES), name).toEqual([
        {
          tool: name,
          param: null,
          kind: "missing-ts-tool",
          observed: `python=read (hint) ${params}`,
        },
      ]);
    }
  });

  it("keeps such a name when the manifest is read as a surface", () => {
    const tools = INHERITED.map((name) => ({
      name,
      tags: ["read"],
      annotations: { readOnlyHint: true },
      inputSchema: { type: "object", properties: { value: { type: "string" } } },
    }));
    const manifest = { source: "x", toolCount: tools.length, tools } as unknown as Manifest;

    expect([...pythonSurface(manifest).keys()]).toEqual(INHERITED);
  });

  it("reports a parameter only one side declares, whatever it is called", () => {
    // Compared under the names the surfaces write, because the snake-case rule
    // renames most of these and what is at stake here is the lookup, not the spelling.
    const asWritten: Normalization = { ...RULES, snakeCaseParamNames: false };
    for (const name of INHERITED) {
      // JSON.parse writes `__proto__` as an ordinary key; an object literal would
      // have set the prototype and left no parameter behind to compare.
      const properties = JSON.parse(`{${JSON.stringify(name)}: {"type": "string"}}`);
      const python = { example: { inputSchema: { type: "object", properties }, annotations: {} } };
      const typescript = {
        example: { inputSchema: { type: "object", properties: {} }, annotations: {} },
      };

      expect(compare(python, typescript, asWritten), name).toEqual([
        {
          tool: "example",
          param: name,
          kind: "missing-ts-param",
          observed: "python=type=string required=false default=none",
        },
      ]);
      // And under the switches CI runs with, whichever spelling they leave.
      expect(compare(python, typescript, RULES).map((d) => d.kind), name).toEqual([
        "missing-ts-param",
      ]);
    }
  });

  it("reads such a name written as a schema keyword as the keyword it is", () => {
    const value = (schema: JsonSchema) => ({
      example: { inputSchema: { type: "object", properties: { value: schema } }, annotations: {} },
    });

    for (const name of INHERITED) {
      // Nothing models these, so each rides along as structure like any other
      // keyword nobody has classified -- none may be read as a keyword shape.
      const carried = JSON.parse(`{"type": "number", ${JSON.stringify(name)}: 1}`);

      expect(unknownKeywords(carried, "example"), name).toEqual([
        { where: "example", keyword: name },
      ]);
      expect(compare(value(carried), value({ type: "number" }), RULES), name).toContainEqual(
        expect.objectContaining({
          kind: "type-mismatch",
          observed: `python=also({${JSON.stringify(name)}:1})&number typescript=number`,
        }),
      );
    }
  });
});

describe("a list of schemas where one is expected", () => {
  const tool = (value: Record<string, unknown>): Record<string, ToolContract> => ({
    example: {
      inputSchema: { type: "object", properties: { value } },
      annotations: {},
      tags: ["read"],
    },
  });

  it("stops the run on tuple items", () => {
    expect(() =>
      compare(
        tool({ type: "array", items: [{ type: "object", default: { description: "a" } }] }),
        {},
        RULES,
      ),
    ).toThrow(/example \(python\).*"items"/s);
  });

  it("looks inside one, so nothing in it escapes the keyword walk", () => {
    const strays = unknownKeywords(
      { type: "array", items: [{ type: "object", "x-galaxy-step": 1 }] },
      "example",
    );
    expect(strays).toEqual([{ where: "example.items[0]", keyword: "x-galaxy-step" }]);
  });

  it("stops the run on prefixItems", () => {
    expect(() =>
      compare(tool({ type: "array", prefixItems: [{ type: "string" }] }), {}, RULES),
    ).toThrow(/example \(python\).*"prefixItems"/s);
  });
});

describe("optional parameters typed `T | None`", () => {
  const withEnum = (value: string) => ({
    type: "object",
    properties: {
      value: { anyOf: [{ type: "string" }, { type: "null" }], enum: [value] },
    },
  });

  it("keeps the keywords beside the union when it folds the null away", () => {
    const python: Record<string, ToolContract> = {
      example: { inputSchema: withEnum("a"), annotations: {} },
    };
    const typescript: Record<string, ToolContract> = {
      example: { inputSchema: withEnum("b"), annotations: {} },
    };

    expect(compare(python, typescript, RULES)).toContainEqual(
      expect.objectContaining({
        kind: "type-mismatch",
        observed: 'python=enum<"a">&string typescript=enum<"b">&string',
      }),
    );
  });

  it("keeps the union when the branch would overwrite a keyword beside it", () => {
    const python: Record<string, ToolContract> = {
      example: {
        inputSchema: {
          type: "object",
          properties: {
            value: { anyOf: [{ type: "string" }, { type: "null" }], type: "object" },
          },
        },
        annotations: {},
      },
    };
    const typescript: Record<string, ToolContract> = {
      example: {
        inputSchema: { type: "object", properties: { value: { type: "string" } } },
        annotations: {},
      },
    };

    expect(compare(python, typescript, RULES)).toContainEqual(
      expect.objectContaining({
        kind: "type-mismatch",
        observed: "python=anyOf<string>&object typescript=string",
      }),
    );
  });
});

describe("schema type normalization", () => {
  it("keeps an anyOf schema's sibling type in the comparison", () => {
    const schema = {
      anyOf: [{ type: "string" }, { type: "integer" }],
      type: "object",
    };
    const python: Record<string, ToolContract> = {
      example: {
        inputSchema: { type: "object", properties: { value: { ...schema, type: "string" } } },
        annotations: {},
      },
    };
    const typescript: Record<string, ToolContract> = {
      example: {
        inputSchema: { type: "object", properties: { value: { ...schema, type: "integer" } } },
        annotations: {},
      },
    };

    expect(compare(python, typescript, RULES)).toContainEqual(
      expect.objectContaining({
        kind: "type-mismatch",
        observed: "python=anyOf<integer|string>&string typescript=anyOf<integer|string>&integer",
      }),
    );
  });
});

describe("an element type, wherever it sits", () => {
  const listOrString = (element: string) => ({
    type: "object",
    properties: {
      value: { anyOf: [{ type: "array" }, { type: "string" }], items: { type: element } },
    },
  });

  it("is read when the array is one branch of a union", () => {
    const python: Record<string, ToolContract> = {
      example: { inputSchema: listOrString("integer"), annotations: {} },
    };
    const typescript: Record<string, ToolContract> = {
      example: { inputSchema: listOrString("string"), annotations: {} },
    };

    expect(compare(python, typescript, RULES)).toContainEqual(
      expect.objectContaining({
        kind: "type-mismatch",
        observed:
          "python=anyOf<array|string>&items<integer> " +
          "typescript=anyOf<array|string>&items<string>",
      }),
    );
  });

  it("is read the same way when the type sits right beside it", () => {
    expect(typeToken({ type: "array", items: { type: "string" } })).toBe("array&items<string>");
    expect(typeToken({ items: { type: "string" } })).toBe("items<string>");
  });
});

describe("schema keywords the type token does not model", () => {
  const value = (schema: Record<string, unknown>) => ({
    type: "object",
    properties: { value: schema },
  });

  it("compares an unmodeled keyword structurally instead of dropping it", () => {
    const python: Record<string, ToolContract> = {
      example: {
        inputSchema: value({ type: "number", "x-galaxy-step": 1 }),
        annotations: {},
      },
    };
    const typescript: Record<string, ToolContract> = {
      example: { inputSchema: value({ type: "number" }), annotations: {} },
    };

    expect(compare(python, typescript, RULES)).toContainEqual(
      expect.objectContaining({
        kind: "type-mismatch",
        observed: 'python=also({"x-galaxy-step":1})&number typescript=number',
      }),
    );
  });

  it("carries a keyword it knows but cannot read", () => {
    expect(typeToken({ type: "" })).not.toBe(typeToken({}));
    expect(typeToken({ type: "" })).toBe('also({"type":""})');
  });

  it("still ignores the value constraints it says it ignores", () => {
    const python: Record<string, ToolContract> = {
      example: { inputSchema: value({ type: "integer", minimum: 1 }), annotations: {} },
    };
    const typescript: Record<string, ToolContract> = {
      example: { inputSchema: value({ type: "integer" }), annotations: {} },
    };

    expect(compare(python, typescript, RULES)).toEqual([]);
  });
});

describe("the generated Python surface manifest", () => {
  it("is internally consistent", () => {
    const names = manifest.tools.map((t) => t.name);
    expect(names.length).toBeGreaterThan(0);
    expect(manifest.toolCount).toBe(names.length);
    expect(names).toEqual([...new Set(names)].sort());
  });

  it("is refused when it does not record what a tool advertises", () => {
    const tools = [{ name: "example", tags: ["read"], inputSchema: { type: "object" } }];
    const stripped = { source: "x", toolCount: 1, tools } as unknown as Manifest;
    expect(() => pythonSurface(stripped)).toThrow(/does not say what example advertises/);
  });

  it("is refused when a tool's tags are not a list of tags", () => {
    const tools = [
      { name: "example", tags: "read", annotations: {}, inputSchema: { type: "object" } },
    ];
    const mistyped = { source: "x", toolCount: 1, tools } as unknown as Manifest;
    expect(() => pythonSurface(mistyped)).toThrow(/"tags" is string/);
  });

  it("tags every tool as read or write, which the comparison relies on", () => {
    const untagged = manifest.tools.filter(
      (t) => !t.tags.includes("read") && !t.tags.includes("write"),
    );
    expect(untagged.map((t) => t.name)).toEqual([]);
  });
});

describe("the keywords the two surfaces use", () => {
  it("are all either modeled or deliberately ignored, none of them new", () => {
    const strays = [
      ...manifest.tools.map((t) => [`python ${t.name}`, t.inputSchema] as const),
      ...[...advertised].map(([name, c]) => [`typescript ${name}`, c.inputSchema] as const),
    ].flatMap(([where, schema]) => unknownKeywords(schema, where));

    expect(
      strays.map((s) => `${s.where}: ${s.keyword}`),
      "a schema keyword nobody has classified: teach the comparison what it means, " +
        "ignore it on purpose, or accept that it is only compared as structure",
    ).toEqual([]);
  });
});

describe("the tools this package advertises", () => {
  it("are exactly the registered ops, with nothing dropped on the way to MCP", () => {
    expect([...advertised.keys()].sort()).toEqual([...toolNames()].sort());
  });

  it("all carry a read-only hint, which is the only thing that says they do not write", () => {
    const unhinted = [...advertised]
      .filter(([, c]) => typeof c.annotations.readOnlyHint !== "boolean")
      .map(([name]) => name);
    expect(
      unhinted,
      "a tool with no readOnlyHint tells a client it may change anything",
    ).toEqual([]);
  });
});

describe("what each side says about changing things", () => {
  const tool = (c: Partial<ToolContract>): Record<string, ToolContract> => ({
    example: { inputSchema: { type: "object" }, annotations: {}, ...c },
  });

  it("reads a tool with no read-only hint as mutating, the way MCP does", () => {
    const found = compare(tool({ tags: ["read"] }), tool({}), RULES);
    expect(found).toContainEqual(
      expect.objectContaining({
        kind: "mutability-mismatch",
        observed: "python=read (tag) typescript=write (mcp default)",
      }),
    );
  });

  it("believes an advertised hint over a tag that disagrees with it", () => {
    const found = compare(
      tool({ tags: ["write"], annotations: { readOnlyHint: true } }),
      tool({ annotations: { readOnlyHint: false } }),
      RULES,
    );
    expect(found).toContainEqual(
      expect.objectContaining({
        kind: "mutability-mismatch",
        observed: "python=read (hint) typescript=write (hint)",
      }),
    );
  });

  it("does not invent a divergence when both sides agree by different means", () => {
    const found = compare(
      tool({ tags: ["write"] }),
      tool({ annotations: { readOnlyHint: false } }),
      RULES,
    );
    expect(found).toEqual([]);
  });
});

describe("what each side says it needs from the server", () => {
  const tool = (requires?: { galaxy: string }): Record<string, ToolContract> => ({
    example: { inputSchema: { type: "object" }, annotations: { readOnlyHint: true }, requires },
  });

  it("reports a bound only one side declares", () => {
    expect(compare(tool({ galaxy: ">=26.1" }), tool(), RULES)).toEqual([
      {
        tool: "example",
        param: null,
        kind: "requires-mismatch",
        observed: "python=>=26.1 typescript=none",
      },
    ]);
  });

  it("reports two bounds that do not read the same", () => {
    expect(compare(tool({ galaxy: ">=26.1" }), tool({ galaxy: ">=26.2" }), RULES)).toContainEqual(
      expect.objectContaining({
        kind: "requires-mismatch",
        observed: "python=>=26.1 typescript=>=26.2",
      }),
    );
  });

  it("does not invent one when both sides declare the same bound", () => {
    expect(compare(tool({ galaxy: ">=26.1" }), tool({ galaxy: ">=26.1" }), RULES)).toEqual([]);
  });

  it("does not invent one over a spelling both surfaces read the same way", () => {
    // `>= 26.1` is legal on both sides and is the same bound to every reader either
    // one has, so it is not a difference; the report would otherwise say "mismatch"
    // beside two columns that read alike.
    expect(compare(tool({ galaxy: ">= 26.1" }), tool({ galaxy: ">=26.1" }), RULES)).toEqual([]);
    expect(compare(tool({ galaxy: ">=26.01" }), tool({ galaxy: ">=26.1" }), RULES)).toEqual([]);
  });

  it("writes the bound in one spelling, whichever the surface used", () => {
    expect(compare(tool({ galaxy: ">=  26.1" }), tool(), RULES)).toEqual([
      expect.objectContaining({ observed: "python=>=26.1 typescript=none" }),
    ]);
    expect(compare(tool({ galaxy: ">=026.01" }), tool(), RULES)).toEqual([
      expect.objectContaining({ observed: "python=>=26.1 typescript=none" }),
    ]);
  });

  it("reports two bounds that differ only past the reach of a JavaScript number", () => {
    // Python reads a component with int() and keeps every digit of it. Read as a
    // number, these two are one value, and the comparison would report agreement
    // between two servers that are not the same server.
    const [older, newer] = [">=26.9007199254740992", ">=26.9007199254740993"];
    expect(compare(tool({ galaxy: newer }), tool({ galaxy: older }), RULES)).toEqual([
      expect.objectContaining({
        kind: "requires-mismatch",
        observed: `python=${newer} typescript=${older}`,
      }),
    ]);
  });

  it("records the bound in what a tool only one side has advertises", () => {
    expect(compare(tool({ galaxy: ">=26.1" }), {}, RULES)).toEqual([
      expect.objectContaining({
        kind: "missing-ts-tool",
        observed: "python=read (hint) requires=>=26.1 params=[]",
      }),
    ]);
  });

  it("stops the run on a bound it cannot write down", () => {
    expect(() => compare(tool({ galaxy: "26.1" }), {}, RULES)).toThrow(
      /example \(python\).*">=MAJOR\.MINOR"/s,
    );
  });

  it("stops the run on a requirement that is about something else", () => {
    const elsewhere = {
      example: { inputSchema: { type: "object" }, annotations: {}, requires: {} },
    } as unknown as Record<string, ToolContract>;
    expect(() => compare(elsewhere, {}, RULES)).toThrow(
      /example \(python\).*says nothing about Galaxy/s,
    );
  });

  it("stops the run on a second requirement beside the Galaxy one", () => {
    // A requirement on one side only is the divergence this reads for, so one it
    // cannot read has to stop the run rather than compare equal to its absence.
    const alsoNeeds = (requires: Record<string, string>) =>
      ({
        example: { inputSchema: { type: "object" }, annotations: { readOnlyHint: true }, requires },
      }) as unknown as Record<string, ToolContract>;

    expect(() =>
      compare(alsoNeeds({ galaxy: ">=26.1", bioblend: ">=1.7" }), alsoNeeds({ galaxy: ">=26.1" }), RULES),
    ).toThrow(/example \(python\).*"bioblend".*only reads what a tool needs from Galaxy/s);
  });

  it("has no bound the two surfaces disagree about today", () => {
    // A statement about today, not a rule. A real disagreement goes in the registry
    // like any other, with a status and a reason, and then this number changes -- the
    // check that a divergence has to be registered is the one above, not this one.
    const declared = [...pythonSurface(manifest)].filter(([, c]) => c.requires);
    expect(
      declared.length,
      "no tool in the manifest declares a requirement, so this test proves nothing",
    ).toBeGreaterThan(0);
    expect(registry.divergences.filter((d) => d.kind === "requires-mismatch").map(where)).toEqual(
      [],
    );
    expect(found.filter((d) => d.kind === "requires-mismatch").map(formatDivergence)).toEqual([]);
  });

  it("sees a bound one of the real surfaces stops declaring", () => {
    const python = new Map(pythonSurface(manifest));
    const declared = [...python]
      .filter(([, contract]) => contract.requires)
      .sort(([a], [b]) => (a < b ? -1 : 1));
    const [name, contract] = declared[0] ?? [];
    expect(name, "no tool in the manifest declares a requirement").toBeDefined();
    const { requires, ...without } = contract as ToolContract;
    python.set(name as string, without);

    const now = compareSurfaces(python, advertised, RULES);
    const gone = `${name} [requires-mismatch] python=none typescript=${requires?.galaxy}`;
    expect(now.filter((d) => d.kind === "requires-mismatch").map(formatDivergence)).toEqual([gone]);

    // And a difference like that one is accepted the way every other one is: register
    // it and the check that everything found is registered has nothing left to say.
    const accepted = new Set(
      [
        ...registry.divergences,
        { tool: name as string, param: null, kind: "requires-mismatch" as const },
      ].map(divergenceKey),
    );
    expect(now.filter((d) => !accepted.has(divergenceKey(d))).map(formatDivergence)).toEqual([]);
  });
});
