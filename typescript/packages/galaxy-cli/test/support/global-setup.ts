// One parent temp directory for the whole run, removed at the end. Each test file gets a
// subdirectory of it from `safe-env.ts`, so nothing is left behind even when a worker is killed
// (a worker process never runs `process.on("exit")` handlers).
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { LEAKY_ENV } from "./leaky-env.js";

/**
 * Runs once in the parent process, before vitest starts its workers. It returns the teardown that
 * removes the run's directory, which runs at the end even when a worker was killed.
 */
export default function setup(): () => void {
  const root = mkdtempSync(join(tmpdir(), "gxy-test-home-"));
  // Workers are started after this, so they inherit it, and they inherit the proxy and TLS
  // variables cleared here too. That has to happen here: Node reads NODE_USE_ENV_PROXY once, at
  // startup, so a worker that was started with it set has already acted on it by the time any
  // setup file runs, and deleting it there changes nothing.
  process.env.GXY_TEST_ROOT = root;
  // safe-env.ts clears GALAXY_URL and GALAXY_API_KEY in every test file, which would switch off the
  // opt-in smoke test (test/bin.smoke.int.test.ts, which runs against a real Galaxy when both are
  // set). It reads these copies instead.
  if (process.env.GALAXY_URL) process.env.GXY_LIVE_URL = process.env.GALAXY_URL;
  if (process.env.GALAXY_API_KEY) process.env.GXY_LIVE_API_KEY = process.env.GALAXY_API_KEY;
  for (const name of LEAKY_ENV) Reflect.deleteProperty(process.env, name);
  return () => {
    rmSync(root, { recursive: true, force: true });
  };
}
