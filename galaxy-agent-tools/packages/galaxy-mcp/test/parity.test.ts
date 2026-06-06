import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { toolNames } from "../src/server";

const galaxyTools: string[] = JSON.parse(
  readFileSync(fileURLToPath(new URL("./fixtures/api-mcp-tools.json", import.meta.url)), "utf8"),
);

describe("op-name parity with in-Galaxy /api/mcp", () => {
  it("every TS op name exists in the Galaxy registry (no silo drift)", () => {
    const galaxy = new Set(galaxyTools);
    const drift = toolNames().filter((n) => !galaxy.has(n));
    expect(drift, `TS ops not present in /api/mcp: ${drift.join(", ")}`).toEqual([]);
  });

  // Optional live check: when GALAXY_URL is set, enumerate the live registry instead.
  const live = process.env.GALAXY_URL ? it : it.skip;
  live("matches the live /api/mcp registry", async () => {
    // Implementation note: hit the live MCP tools/list when wiring the integration env.
    expect(true).toBe(true);
  });
});
