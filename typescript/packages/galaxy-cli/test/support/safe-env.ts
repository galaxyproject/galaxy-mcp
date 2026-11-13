// A safety net for every test file: whatever a test forgets to inject, the real keyring, the real
// config directory and the developer's own settings stay out of reach. The keyring is off, the
// config dir and HOME are an empty temp directory, and every variable the code reads, or Node reads
// for it (the proxy and TLS ones, see leaky-env.ts), is cleared.
// Tests that need something else set it themselves (vi.stubEnv, or an `env` they pass in), and
// the one test that runs the real keyring child does so against a fake bus.
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
// All three are pointed at the temp directory because the code reads different ones: the config
// directory comes from `GXY_CONFIG_DIR`, else `XDG_CONFIG_HOME`, else HOME, and the planemo profile
// is looked up under HOME. Setting only one would leave a path to the real files.
process.env.HOME = home;
process.env.XDG_CONFIG_HOME = join(home, "xdg");
process.env.GXY_CONFIG_DIR = join(home, "gxy");
// With the keyring off a store that is built from the environment keeps tokens in the file, and
// never starts a child that could reach a real keyring.
process.env.GXY_NO_KEYRING = "1";

// Everything under src/ that reads the environment, so a developer's own values cannot change a result.
// The live values were saved by global-setup.ts for the opt-in smoke test.
for (const name of [
  "DBUS_SESSION_BUS_ADDRESS",
  "GALAXY_URL",
  "GALAXY_API_KEY",
  "CLAUDE_CODE_MCP_SERVER_URL",
  "GXY_OAUTH_CLIENT_ID",
  "BROWSER",
  "DISPLAY",
  "WAYLAND_DISPLAY",
  ...LEAKY_ENV,
]) {
  Reflect.deleteProperty(process.env, name);
}

afterAll(() => {
  rmSync(home, { recursive: true, force: true });
});
