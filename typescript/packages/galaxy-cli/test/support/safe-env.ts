// A safety net for every test file: whatever a test forgets to inject, the developer's own home
// directory and settings stay out of reach. HOME is an empty temp directory (the planemo profile
// is looked up under it), and every variable the code reads, or Node reads for it (the proxy and
// TLS ones, see leaky-env.ts), is cleared.
// Tests that need something else set it themselves (vi.stubEnv, or an `env` they pass in).
//
// Not covered here: the working directory, which `loadSources` reads `.env` from. A test that
// runs it spies on `process.cwd` or runs a child in a temp cwd.
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll } from "vitest";
import { LEAKY_ENV } from "./leaky-env.js";

// Each test file gets its own directory, under the run's parent directory (global-setup.ts, which
// removes the parent at the end). The fallback covers running this file without it. It is removed
// in `afterAll`, not `process.on("exit")`, because a vitest worker never runs exit handlers.
const home = mkdtempSync(join(process.env.GXY_TEST_ROOT ?? tmpdir(), "gxy-test-home-"));
process.env.HOME = home;

// Everything under src/ that reads the environment, so a developer's own values cannot change a result.
// global-setup.ts passes the live values to the opt-in smoke test through `provide`.
for (const name of ["GALAXY_URL", "GALAXY_API_KEY", ...LEAKY_ENV]) {
  Reflect.deleteProperty(process.env, name);
}

afterAll(() => {
  rmSync(home, { recursive: true, force: true });
});
