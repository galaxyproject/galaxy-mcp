import { beforeAll, describe, expect, it } from "vitest";
import { toolNames } from "../src/server";
import {
  compareSurfaces,
  divergenceKey,
  formatDivergence,
  NORMALIZATION_RULES,
  WHOLE_TOOL_KINDS,
  type Divergence,
  type Normalization,
  type ToolContract,
} from "./parity/compare";
import {
  DIVERGENCE_STATUSES,
  loadManifest,
  loadRegistry,
  normalizationFrom,
  pythonSurface,
  typescriptSurface,
  type Manifest,
  type Registry,
} from "./parity/surfaces";

const KINDS = [
  "missing-ts-tool",
  "missing-py-tool",
  "missing-ts-param",
  "missing-py-param",
  "type-mismatch",
  "required-mismatch",
  "default-mismatch",
  "mutability-mismatch",
];

const where = (d: { tool: string; param: string | null; kind: string }) =>
  `${d.tool}${d.param ? `.${d.param}` : ""} [${d.kind}]`;

let manifest: Manifest;
let registry: Registry;
let advertised: Record<string, ToolContract>;
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
      if (!KINDS.includes(entry.kind)) problems.push(`${where(entry)}: unknown kind`);
      if (!DIVERGENCE_STATUSES.includes(entry.status)) {
        problems.push(`${where(entry)}: unknown status`);
      }
      if (!entry.reason?.trim()) problems.push(`${where(entry)}: needs a reason`);
      if (typeof entry.observed !== "string") {
        problems.push(`${where(entry)}: needs an observed string`);
      }
      if (WHOLE_TOOL_KINDS.includes(entry.kind) !== (entry.param === null)) {
        problems.push(`${where(entry)}: param must be null for whole-tool kinds and set otherwise`);
      }
      const key = divergenceKey(entry);
      if (seen.has(key)) problems.push(`${where(entry)}: duplicate entry`);
      seen.add(key);
    }
    expect(problems).toEqual([]);
  });

  it("declares every normalization rule the comparator applies", () => {
    const declared = Object.keys(registry.normalization).sort();
    expect(
      declared,
      "the registry must declare exactly the rules the comparator knows about",
    ).toEqual([...NORMALIZATION_RULES].sort());
    for (const name of NORMALIZATION_RULES) {
      const rule = registry.normalization[name];
      expect(typeof rule.enabled, `${name}.enabled`).toBe("boolean");
      expect(DIVERGENCE_STATUSES, `${name}.status`).toContain(rule.status);
      expect(rule.reason.trim().length, `${name}.reason`).toBeGreaterThan(0);
    }
  });

  it("covers every kind the comparator can report", () => {
    expect([...KINDS].sort()).toEqual([...new Set(KINDS)].sort());
    for (const kind of WHOLE_TOOL_KINDS) expect(KINDS).toContain(kind);
  });
});

describe("missing parameter divergences", () => {
  const rules: Normalization = {
    snakeCaseParamNames: false,
    pythonNullDefaults: false,
    optionalNullUnions: false,
  };

  it("makes a registered missing parameter stale when its existing contract changes", () => {
    const typescript: Record<string, ToolContract> = {
      example: { inputSchema: { type: "object" }, mutating: false },
    };
    const accepted = "python=type=integer required=false default=10";
    const example: ToolContract = {
      inputSchema: {
        type: "object",
        properties: { preview_lines: { default: 10, type: "integer" } },
      },
      mutating: false,
    };
    const python: Record<string, ToolContract> = { example };

    expect(compareSurfaces(python, typescript, rules)).toContainEqual(
      expect.objectContaining({ observed: accepted }),
    );

    example.inputSchema.properties!.preview_lines = {
      default: "all",
      type: "string",
    };
    example.inputSchema.required = ["preview_lines"];

    const changed = compareSurfaces(python, typescript, rules);
    expect(changed).toContainEqual(
      expect.objectContaining({
        observed: "python=type=string required=true default=\"all\"",
      }),
    );
    expect(changed.map((d) => d.observed)).not.toContain(accepted);
  });
});

describe("schema type normalization", () => {
  const rules: Normalization = {
    snakeCaseParamNames: false,
    pythonNullDefaults: false,
    optionalNullUnions: false,
  };

  it("keeps an anyOf schema's sibling type in the comparison", () => {
    const schema = {
      anyOf: [{ type: "string" }, { type: "integer" }],
      type: "object",
    };
    const python: Record<string, ToolContract> = {
      example: {
        inputSchema: { properties: { value: { ...schema, type: "string" } } },
        mutating: false,
      },
    };
    const typescript: Record<string, ToolContract> = {
      example: {
        inputSchema: { properties: { value: { ...schema, type: "integer" } } },
        mutating: false,
      },
    };

    expect(compareSurfaces(python, typescript, rules)).toContainEqual(
      expect.objectContaining({
        kind: "type-mismatch",
        observed: "python=anyOf<integer|string>&string typescript=anyOf<integer|string>&integer",
      }),
    );
  });
});

describe("the generated Python surface manifest", () => {
  it("is internally consistent", () => {
    const names = manifest.tools.map((t) => t.name);
    expect(names.length).toBeGreaterThan(0);
    expect(manifest.toolCount).toBe(names.length);
    expect(names).toEqual([...new Set(names)].sort());
  });

  it("tags every tool as read or write, which the comparison relies on", () => {
    const untagged = manifest.tools.filter(
      (t) => !t.tags.includes("read") && !t.tags.includes("write"),
    );
    expect(untagged.map((t) => t.name)).toEqual([]);
  });
});

describe("the tools this package advertises", () => {
  it("are exactly the registered ops, with nothing dropped on the way to MCP", () => {
    expect(Object.keys(advertised).sort()).toEqual([...toolNames()].sort());
  });
});
