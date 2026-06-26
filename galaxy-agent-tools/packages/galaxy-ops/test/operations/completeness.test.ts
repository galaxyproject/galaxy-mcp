import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import "../../src/operations/all";
import { allOperations } from "../../src/operations/registry";

const INTENTIONAL_GAPS: Set<string> = new Set(JSON.parse(
  readFileSync(fileURLToPath(new URL("../../../galaxy-mcp/test/fixtures/intentional-gaps.json", import.meta.url)), "utf8"),
));

describe("registry completeness", () => {
  it("registered op names exactly match the external parity fixture (no missing, no extra)", () => {
    const fixture: string[] = JSON.parse(
      readFileSync(fileURLToPath(new URL("../../../galaxy-mcp/test/fixtures/external-mcp-tools.json", import.meta.url)), "utf8"),
    );
    const fixtureSet = new Set(fixture);
    const operations = allOperations.map((o) => o.name);
    const operationSet = new Set(operations);
    const opsDrift = operations.filter((n) => !fixtureSet.has(n));
    expect(opsDrift, `ops not in fixture: ${opsDrift.join(", ")}`).toEqual([]);
    const fixtureDrift = fixture.filter((n) => !operationSet.has(n) && !INTENTIONAL_GAPS.has(n));
    expect(fixtureDrift, `fixture names not in operations list: ${fixtureDrift.join(", ")}`).toEqual([]);
  });
});
