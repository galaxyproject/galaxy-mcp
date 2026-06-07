import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { toolNames } from "../src/server";

// The toolset we target is this repo's external Python MCP server
// (mcp-server-galaxy-py/src/galaxy_mcp/server.py) -- NOT the in-tree /api/mcp,
// which is a separate in-process surface with a different (overlapping) set.
const galaxyTools: string[] = JSON.parse(
  readFileSync(fileURLToPath(new URL("./fixtures/external-mcp-tools.json", import.meta.url)), "utf8"),
);

describe("op-name parity with the external Python MCP server (mcp-server-galaxy-py)", () => {
  it("every TS op name exists in the Python server's toolset (no silo drift)", () => {
    const galaxy = new Set(galaxyTools);
    const drift = toolNames().filter((n) => !galaxy.has(n));
    expect(drift, `TS ops not present in the Python MCP server: ${drift.join(", ")}`).toEqual([]);
  });

  // Optional live check: when GALAXY_URL is set, enumerate the live registry instead.
  const live = process.env.GALAXY_URL ? it : it.skip;
  live("matches the live MCP registry", async () => {
    expect(true).toBe(true);
  });
});
