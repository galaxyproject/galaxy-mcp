import { describe, it, expect, beforeAll } from "vitest";
import { execa } from "execa";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

// Builds into a temporary directory (so it doesn't touch dist/; inside the package, so node finds its dependencies) and runs both bins' --help. It
// needs no Galaxy credentials. A wrong tsup entry would ship a bin name that doesn't exist.
const pkgDir = fileURLToPath(new URL("..", import.meta.url));
let outDir: string;

beforeAll(async () => {
  // The bins import the workspace package galaxy-ops from its dist, so that has to exist.
  await execa("pnpm", ["--filter", "@galaxyproject/galaxy-ops", "build"], { cwd: pkgDir });
  outDir = mkdtempSync(join(pkgDir, ".gxy-bins-"));
  await execa("pnpm", ["exec", "tsup", "--out-dir", outDir], { cwd: pkgDir });
  return () => rmSync(outDir, { recursive: true, force: true });
}, 120_000);

describe("built bins", () => {
  it.each([
    ["gxy.js", "gxy"],
    ["index.js", "galaxy-cli"],
  ])("%s --help prints a usage line naming %s", async (file, name) => {
    const { stdout, exitCode } = await execa("node", [join(outDir, file), "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toMatch(new RegExp(`^Usage: ${name} `, "m"));
  }, 30_000);
});
