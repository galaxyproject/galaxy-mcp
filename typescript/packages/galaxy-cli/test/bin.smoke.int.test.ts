import { describe, it, expect, inject } from "vitest";
import { execa } from "execa";
import { fileURLToPath } from "node:url";

const bin = fileURLToPath(new URL("../dist/index.js", import.meta.url));
// The test setup clears GALAXY_URL and GALAXY_API_KEY from the environment; global-setup.ts passes
// the values you set to this file alone, through vitest's `provide`/`inject`.
const { url: BASE_URL, apiKey: KEY } = inject("gxyLive");
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
