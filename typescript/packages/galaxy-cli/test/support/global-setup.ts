// One parent temp directory for the whole run, removed at the end. Each test file gets a
// subdirectory of it from `safe-env.ts`, so nothing is left behind even when a worker is killed
// (a worker process never runs `process.on("exit")` handlers).
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { TestProject } from "vitest/node";
import { LEAKY_ENV } from "./leaky-env.js";

declare module "vitest" {
  export interface ProvidedContext {
    /** The developer's GALAXY_URL and GALAXY_API_KEY, for the opt-in smoke test only. */
    gxyLive: { url?: string; apiKey?: string };
  }
}

/**
 * Runs once in the parent process, before vitest starts its workers. It returns the teardown that
 * removes the run's directory, which runs at the end even when a worker was killed.
 */
export default function setup(project: Pick<TestProject, "provide">): () => void {
  const root = mkdtempSync(join(tmpdir(), "gxy-test-home-"));
  // Workers are started after this, so they inherit it, and they inherit the proxy and TLS
  // variables cleared here too. That has to happen here: Node reads NODE_USE_ENV_PROXY once, at
  // startup, so a worker that was started with it set has already acted on it by the time any
  // setup file runs, and deleting it there changes nothing.
  process.env.GXY_TEST_ROOT = root;
  // safe-env.ts clears GALAXY_URL and GALAXY_API_KEY in every test file, which would switch off the
  // opt-in smoke test (test/bin.smoke.int.test.ts, which runs against a real Galaxy when both are
  // set). It reads them with `inject("gxyLive")` instead. vitest's `provide` hands the values to
  // each test file directly, rather than through process.env, which every worker (and every child
  // process a test spawns with the default env) would inherit, live key included.
  // GXY_LIVE_URL and GXY_LIVE_API_KEY, if set, are accepted as the same thing under their own
  // names, and removed from the environment here too, so no worker inherits either.
  project.provide("gxyLive", {
    url: process.env.GXY_LIVE_URL || process.env.GALAXY_URL || undefined,
    apiKey: process.env.GXY_LIVE_API_KEY || process.env.GALAXY_API_KEY || undefined,
  });
  Reflect.deleteProperty(process.env, "GXY_LIVE_URL");
  Reflect.deleteProperty(process.env, "GXY_LIVE_API_KEY");
  for (const name of LEAKY_ENV) Reflect.deleteProperty(process.env, name);
  return () => {
    rmSync(root, { recursive: true, force: true });
  };
}
