import { beforeAll, describe, expect, it } from "vitest";
import { toolNames } from "../src/server";
import {
  compareSurfaces,
  describe as describeDivergence,
  divergenceKey,
  type Divergence,
} from "./parity/compare";
import {
  DIVERGENCE_STATUSES,
  loadManifest,
  loadRegistry,
  normalizationFrom,
  pythonSurface,
  typescriptSurface,
  type AcceptedDivergence,
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
];
const WHOLE_TOOL_KINDS = new Set(["missing-ts-tool", "missing-py-tool"]);

const FIX = (extra: string) =>
  `\n\nReview each one and add it to test/fixtures/accepted-divergences.json,${extra}`;

let manifest: Manifest;
let registry: Registry;
let found: Divergence[];

beforeAll(async () => {
  manifest = loadManifest();
  registry = loadRegistry();
  found = compareSurfaces(
    pythonSurface(manifest),
    await typescriptSurface(),
    normalizationFrom(registry),
  );
});

describe("contract parity with the Python MCP server", () => {
  it("has no divergence that is missing from the accepted-divergence registry", () => {
    const accepted = new Set(registry.divergences.map(divergenceKey));
    const unregistered = found.filter((d) => !accepted.has(divergenceKey(d)));
    expect(
      unregistered.map(describeDivergence),
      `${unregistered.length} unregistered divergence(s) between the TS ops and the Python ` +
        `server.${FIX(" or close the gap in the op.")}`,
    ).toEqual([]);
  });

  it("has no registry entry the two surfaces no longer disagree about", () => {
    const current = new Set(found.map(divergenceKey));
    const stale = registry.divergences.filter((d) => !current.has(divergenceKey(d)));
    expect(
      stale.map((d) => `${d.tool}${d.param ? `.${d.param}` : ""} [${d.kind}]`),
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
          `${entry.tool}${entry.param ? `.${entry.param}` : ""} [${entry.kind}]: ` +
          `registry says "${entry.observed}", surfaces say "${actual?.observed}"`,
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
    for (const entry of registry.divergences as AcceptedDivergence[]) {
      const where = `${entry.tool}${entry.param ? `.${entry.param}` : ""} [${entry.kind}]`;
      if (!KINDS.includes(entry.kind)) problems.push(`${where}: unknown kind`);
      if (!DIVERGENCE_STATUSES.includes(entry.status)) problems.push(`${where}: unknown status`);
      if (!entry.reason?.trim()) problems.push(`${where}: needs a reason`);
      if (typeof entry.observed !== "string") problems.push(`${where}: needs an observed string`);
      if (WHOLE_TOOL_KINDS.has(entry.kind) !== (entry.param === null)) {
        problems.push(`${where}: param must be null for whole-tool kinds and set otherwise`);
      }
      const key = divergenceKey(entry);
      if (seen.has(key)) problems.push(`${where}: duplicate entry`);
      seen.add(key);
    }
    expect(problems).toEqual([]);
  });

  it("declares every normalization rule the comparator applies", () => {
    for (const name of ["snakeCaseParamNames", "pythonNullDefaults"] as const) {
      const rule = registry.normalization[name];
      expect(rule, `${name} must be declared`).toBeDefined();
      expect(typeof rule.enabled, `${name}.enabled`).toBe("boolean");
      expect(DIVERGENCE_STATUSES, `${name}.status`).toContain(rule.status);
      expect(rule.reason.trim().length, `${name}.reason`).toBeGreaterThan(0);
    }
  });
});

describe("the generated Python surface manifest", () => {
  it("is internally consistent", () => {
    const names = manifest.tools.map((t) => t.name);
    expect(names.length).toBeGreaterThan(0);
    expect(manifest.toolCount).toBe(names.length);
    expect(names).toEqual([...new Set(names)].sort());
  });

  it("describes an input schema for every tool", () => {
    const bare = manifest.tools.filter((t) => typeof t.inputSchema?.type !== "string");
    expect(bare.map((t) => t.name)).toEqual([]);
  });
});

describe("the tools this package advertises", () => {
  it("are exactly the registered ops", async () => {
    const advertised = Object.keys(await typescriptSurface()).sort();
    expect(advertised).toEqual([...toolNames()].sort());
  });
});
