/**
 * `src/galaxy-mcp-surface.ts` is generated from mcp-server-galaxy-py/tests/testdata/mcp-surface.json,
 * which that server's suite writes off the tools it registers and keeps current
 * (`uv run python -m tests.surface_manifest`, checked by `tests/test_mcp_surface_snapshot.py`).
 * Two checked-in copies can drift, so this reads the original and compares them.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { GALAXY_MCP_SURFACE } from "../src/galaxy-mcp-surface";

const MANIFEST = new URL("../../../../mcp-server-galaxy-py/tests/testdata/mcp-surface.json", import.meta.url);

type Tool = {
  name: string;
  description: string;
  inputSchema?: { properties?: Record<string, { description?: string }> };
};

describe("galaxy-mcp's advertised surface", () => {
  it("is the one its server's manifest records, description and parameters alike", () => {
    const tools: Tool[] = JSON.parse(readFileSync(MANIFEST, "utf8")).tools;
    const expected = Object.fromEntries(
      tools.map((tool) => [
        tool.name,
        {
          description: tool.description,
          parameters: Object.fromEntries(
            Object.entries(tool.inputSchema?.properties ?? {}).map(([name, schema]) => [
              name,
              schema.description ?? "",
            ]),
          ),
        },
      ]),
    );
    expect(GALAXY_MCP_SURFACE).toEqual(expected);
    for (const tool of tools) {
      expect(Object.keys(GALAXY_MCP_SURFACE[tool.name]!.parameters), tool.name).toEqual(
        Object.keys(tool.inputSchema?.properties ?? {}),
      );
    }
  });
});
