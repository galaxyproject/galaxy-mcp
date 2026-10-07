/**
 * Turn the other server's checked-in surface manifest into the module this package exports.
 *
 * Run from packages/galaxy-ops:
 *
 *   node scripts/generate-galaxy-mcp-surface.mjs
 *
 * The input is mcp-server-galaxy-py/tests/testdata/mcp-surface.json, which the Python suite
 * generates off the tools that server registers (`uv run python -m tests.surface_manifest`). The
 * output is src/galaxy-mcp-surface.ts: each tool's advertised description and its parameters'
 * descriptions, checked in so the package needs no Python to build, and compared with the manifest by
 * `test/galaxy-mcp-surface.test.ts` so the two cannot drift.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const MANIFEST = new URL("../../../../mcp-server-galaxy-py/tests/testdata/mcp-surface.json", import.meta.url);
const OUT = new URL("../src/galaxy-mcp-surface.ts", import.meta.url);

const manifest = JSON.parse(readFileSync(MANIFEST, "utf8"));
const entries = manifest.tools.map((tool) => {
  const parameters = Object.entries(tool.inputSchema?.properties ?? {}).map(
    ([name, schema]) => `      ${JSON.stringify(name)}: ${JSON.stringify(schema.description ?? "")},`,
  );
  return (
    `  ${JSON.stringify(tool.name)}: {\n` +
    `    description: ${JSON.stringify(tool.description)},\n` +
    `    parameters: {\n${parameters.join("\n")}\n    },\n` +
    `  },`
  );
});

const text = `/**
 * What galaxy-mcp's Python server advertises for each of its tools, as an MCP client receives it:
 * the description, with any Galaxy requirement in it, and each parameter's description, in the
 * order the tool declares them ("" where it has none). For a client that offers these tools under
 * galaxy-mcp's names and wants its model told what that server's is.
 *
 * GENERATED FILE -- do not edit. Regenerate from packages/galaxy-ops with
 * \`node scripts/generate-galaxy-mcp-surface.mjs\`. Its input is
 * mcp-server-galaxy-py/tests/testdata/mcp-surface.json, which that server writes with
 * \`uv run python -m tests.surface_manifest\`, and \`test/galaxy-mcp-surface.test.ts\` compares this
 * file to that one, so the two copies cannot drift.
 */
export interface GalaxyMcpTool {
  readonly description: string;
  readonly parameters: Readonly<Record<string, string>>;
}

export const GALAXY_MCP_SURFACE: Readonly<Record<string, GalaxyMcpTool>> = {
${entries.join("\n")}
};
`;

writeFileSync(fileURLToPath(OUT), text);
console.log(`wrote src/galaxy-mcp-surface.ts: ${entries.length} tools`);
