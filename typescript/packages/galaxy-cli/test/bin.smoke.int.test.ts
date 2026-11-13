import { describe, it, expect } from "vitest";
import { execa } from "execa";
import { fileURLToPath } from "node:url";

const bin = fileURLToPath(new URL("../dist/index.js", import.meta.url));
// The test setup clears GALAXY_URL and GALAXY_API_KEY; global-setup.ts keeps the values you set.
const BASE_URL = process.env.GXY_LIVE_URL;
const KEY = process.env.GXY_LIVE_API_KEY;
const run = BASE_URL && KEY ? describe : describe.skip;

run("galaxy-cli built bin", () => {
  it("get_user --format json returns the user", async () => {
    const { stdout, exitCode } = await execa("node", [bin, "get_user", "--format", "json"], {
      env: { GALAXY_URL: BASE_URL, GALAXY_API_KEY: KEY },
    });
    expect(exitCode).toBe(0);
    const parsed = JSON.parse(stdout);
    expect(parsed.success).toBe(true);
    expect(parsed.data.id).toBeTruthy();
  }, 30_000);
});
