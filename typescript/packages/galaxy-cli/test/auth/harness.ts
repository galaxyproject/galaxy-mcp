/** Shared fixtures for the credential store's tests: a Galaxy, a login, a temp directory. */
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach } from "vitest";
import { normalizeBaseUrl } from "../../src/auth/base-url";
import type { BaseUrl, StoredLogin } from "../../src/auth/types";

/** A fixed time, so a stored time does not depend on when the suite runs. */
export const DAY = "2026-11-06T10:44:00.000Z";
export const HOST: BaseUrl = normalizeBaseUrl("https://dev.galaxy.example.org");
export const OTHER: BaseUrl = normalizeBaseUrl("https://other.example.org");
/** Obviously fake. */
export const FAKE_TOKEN = "fake-token-0123456789abcdef";

/** A stored pasted-key login for `HOST`, with no expiry; `over` replaces any field. */
export function fakeLogin(over: Partial<StoredLogin> = {}): StoredLogin {
  return {
    baseUrl: HOST,
    username: "aioue",
    token: FAKE_TOKEN,
    method: "pasted",
    scopes: [],
    expiresAt: null,
    obtainedAt: DAY,
    ...over,
  };
}

/** A temp directory per test, removed afterwards. Call it at the top of a file or a `describe`. */
export function useTempDir(prefix = "gxy-test-"): { readonly path: string } {
  let dir = "";
  beforeEach(async () => {
    // Under the run's root (`global-setup.ts` removes it at the end), so a killed worker leaves nothing.
    dir = await mkdtemp(join(process.env.GXY_TEST_ROOT ?? tmpdir(), prefix));
  });
  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });
  return {
    get path() {
      return dir;
    },
  };
}
