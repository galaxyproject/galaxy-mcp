import { mkdir, readFile, writeFile } from "node:fs/promises";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { logoutCommand } from "../../src/auth/commands/logout";
import type { FileStore } from "../../src/auth/file-store";
import type { CombinedStore } from "../../src/auth/store";
import { AuthError } from "../../src/auth/types";
import {
  FAKE_TOKEN,
  HOST,
  FakeKeyring,
  fakeLogin,
  harness,
  makeTokenUnreadable,
  realStore,
  useTempDir,
} from "./harness";

describe("auth logout", () => {
  it("revokes the device token while the login is still stored, then deletes the login", async () => {
    const h = harness({ stored: [fakeLogin()] });
    const revoke = h.server.revoke.bind(h.server);
    h.server.revoke = async (token) => {
      // The token can only be revoked if it has not been deleted yet.
      expect(await h.store.get(HOST)).not.toBeNull();
      await revoke(token);
    };
    expect(await logoutCommand({}, h.deps)).toBe(0);
    expect(h.server.revoked).toEqual([FAKE_TOKEN]);
    expect(await h.store.get(HOST)).toBeNull();
    expect(h.cap.err).toContain("Revoked the token on dev.galaxy.example.org");
    expect(h.cap.out).toBe("");
    expect(h.cap.err).not.toContain(FAKE_TOKEN);
  });

  // Local removal does not wait on the server: a Galaxy that is down is when a person most needs
  // the stored login gone. The token then stays valid until it expires, and the message says so.
  it.each([
    [
      "revocation fails",
      (h: ReturnType<typeof harness>) =>
        void (h.server.revokeError = new AuthError("network", "connection refused")),
      "connection refused",
    ],
    [
      "discovery fails",
      (h: ReturnType<typeof harness>) =>
        void (h.knobs.discoverError = new AuthError("network", "no route")),
      "no route",
    ],
  ])(
    "still deletes locally when %s, and says the token stays valid until expiry",
    async (_when, breakIt, why) => {
      const h = harness({ stored: [fakeLogin()] });
      breakIt(h);
      expect(await logoutCommand({}, h.deps)).toBe(0);
      expect(await h.store.get(HOST)).toBeNull();
      expect(h.cap.err).toContain("Could not revoke the token on the server");
      expect(h.cap.err).toContain(why);
      expect(h.cap.err).toMatch(/stays valid until it expires/);
    },
  );

  // Logout is what a person runs when the network is wrong. Without its own limit it would wait for
  // undici's, minutes later, before the local removal they asked for.
  describe("when the server does not answer", () => {
    const hang = (h: ReturnType<typeof harness>, where: "discovery" | "revocation") => {
      if (where === "discovery") h.deps.discover = () => new Promise(() => {});
      else h.server.revoke = () => new Promise(() => {});
    };

    it.each(["discovery", "revocation"] as const)(
      "gives up on %s after 5 s, deletes locally and says why",
      async (where) => {
        vi.useFakeTimers();
        try {
          const h = harness({ stored: [fakeLogin()] });
          hang(h, where);
          const p = logoutCommand({}, h.deps);
          await vi.advanceTimersByTimeAsync(4900);
          expect(await h.store.get(HOST)).not.toBeNull(); // still waiting
          await vi.advanceTimersByTimeAsync(200);
          expect(await p).toBe(0);
          expect(await h.store.get(HOST)).toBeNull();
          expect(h.cap.err).toContain("Could not revoke the token on the server (timed out");
          expect(h.cap.err).toMatch(/stays valid until it expires/);
        } finally {
          vi.useRealTimers();
        }
      },
    );

    it("tells discovery and the revocation to stop when the time is up", async () => {
      vi.useFakeTimers();
      try {
        const h = harness({ stored: [fakeLogin()] });
        let seen: AbortSignal | undefined;
        h.deps.discover = (_b, signal) => ((seen = signal), new Promise(() => {}));
        const p = logoutCommand({}, h.deps);
        await vi.advanceTimersByTimeAsync(5100);
        await p;
        expect(seen?.aborted).toBe(true);
      } finally {
        vi.useRealTimers();
      }
    });
  });

  // An expired token cannot be used, so there is nothing to take back: no request, and no warning
  // that it "stays valid until" a time that has gone.
  it("does not try to revoke a token that has already expired, nor warn about it", async () => {
    const h = harness({ stored: [fakeLogin({ expiresAt: "2026-11-05T18:42:00.000Z" })] });
    let discovered = 0;
    h.deps.discover = async () => (discovered++, h.server);
    expect(await logoutCommand({}, h.deps)).toBe(0);
    expect(discovered).toBe(0);
    expect(await h.store.get(HOST)).toBeNull();
    expect(h.cap.err).toContain("✓ Removed the stored login for dev.galaxy.example.org");
    expect(h.cap.err).not.toMatch(/stays valid|Could not revoke/);
  });

  it("still tries to revoke a token within its last minute, which has not expired yet", async () => {
    const h = harness({ stored: [fakeLogin({ expiresAt: "2026-11-06T10:44:30.000Z" })] });
    expect(await logoutCommand({}, h.deps)).toBe(0);
    expect(h.server.revoked).toEqual([FAKE_TOKEN]);
  });

  it("says so when the server has no revocation endpoint", async () => {
    const h = harness({ stored: [fakeLogin()] });
    h.server.supportsRevocation = false;
    await logoutCommand({}, h.deps);
    expect(h.cap.err).toContain("does not offer token revocation");
    expect(await h.store.get(HOST)).toBeNull();
  });

  it("deletes a pasted key without contacting the server and says it was not revoked", async () => {
    const h = harness({ stored: [fakeLogin({ method: "pasted", scopes: [], expiresAt: null })] });
    // Only a device token is revoked. A pasted key may be in use elsewhere, so logout only forgets
    // it and says to manage it in Galaxy's user preferences instead.
    // Discovery succeeds here, and its calls are counted: a logout that did try to revoke a pasted
    // key would make them. Making discovery throw would not show that, since logout treats a throw
    // as "could not revoke", never prints that for a pasted key, and revokes nothing.
    let discovered = 0;
    const discover = h.deps.discover.bind(h.deps);
    h.deps.discover = async (...args) => (discovered++, discover(...args));
    expect(await logoutCommand({}, h.deps)).toBe(0);
    expect(discovered).toBe(0);
    expect(h.server.revoked).toEqual([]);
    expect(await h.store.get(HOST)).toBeNull();
    expect(h.cap.out).toBe("");
    expect(h.cap.err).toContain("✓ Removed the stored API key for dev.galaxy.example.org");
    expect(h.cap.err).toContain("not revoked");
    expect(h.cap.err).toContain("user preferences");
    expect(h.cap.err).not.toContain(FAKE_TOKEN);
  });

  // Anything that is not an `AuthError` is a fault in gxy, not something the person can fix.
  it("reports an error it did not expect as unexpected, exit 70", async () => {
    const h = harness({ stored: [fakeLogin()] });
    h.store.list = async () => {
      throw new Error("store is broken");
    };
    expect(await logoutCommand({}, h.deps)).toBe(70);
    expect(h.cap.err).toContain("store is broken");
  });

  // Reading the token is only needed to revoke it, and must not stand between the person and
  // removing the record: the keyring may be down or the secret gone, which is when cleaning up matters most.
  describe("when the token cannot be read", () => {
    it.each(["throw", "null"] as const)(
      "skips revocation, removes the record and says what happened (%s)",
      async (how) => {
        const h = harness({ stored: [fakeLogin()] });
        makeTokenUnreadable(h, [HOST], how);
        expect(await logoutCommand({}, h.deps)).toBe(0);
        expect(h.server.revoked).toEqual([]);
        expect(await h.store.list()).toEqual([]);
        expect(h.cap.err).toContain("Removed the stored login for dev.galaxy.example.org");
        expect(h.cap.err).toMatch(/Could not revoke the token on the server.*could not be read/);
        expect(h.cap.err).toContain("stays valid until it expires (2026-11-06 18:42 UTC)");
        expect(h.cap.err).not.toContain(FAKE_TOKEN);
      },
    );

    it("removes a pasted key's record too, noting that it was not revoked", async () => {
      const h = harness({ stored: [fakeLogin({ method: "pasted", scopes: [], expiresAt: null })] });
      makeTokenUnreadable(h, [HOST]);
      expect(await logoutCommand({}, h.deps)).toBe(0);
      expect(h.cap.err).toContain("not revoked");
      expect(await h.store.list()).toEqual([]);
    });
  });

  it("is a no-op with a note when nothing is stored for the host", async () => {
    const h = harness();
    expect(await logoutCommand({ host: HOST }, h.deps)).toBe(0);
    expect(h.cap.err).toContain("nothing to remove");
  });
});

// The real store over a real `hosts.yml` and a fake keyring, because what logout must leave behind
// (no token in the keyring, none in the file) is what the real store does, not what a fake store says.
describe("logout with the Galaxy down, over the real store", () => {
  const tmp = useTempDir("gxy-lo-");
  let keyring: FakeKeyring;
  let file: FileStore;
  let store: CombinedStore;
  beforeEach(() => {
    keyring = new FakeKeyring();
    ({ file, store } = realStore(tmp.path, keyring));
  });

  function logoutHarness() {
    const h = harness();
    h.deps.store = store;
    h.deps.resolveHost = async () => HOST;
    h.knobs.discoverError = new AuthError("network", "connection refused");
    return h;
  }

  it("deletes the keyring secret and the record, and reports the failed revoke", async () => {
    await store.put(fakeLogin());
    const h = logoutHarness();
    expect(await logoutCommand({}, h.deps)).toBe(0);
    expect(keyring.data.size).toBe(0);
    expect(await file.list()).toEqual([]);
    expect(h.cap.err).toContain("Could not revoke the token on the server");
    expect(h.cap.err).toContain("connection refused");
  });

  // The list leaves out a login this version cannot read, so "nothing to remove" and a 0 would
  // tell the person it is gone while it stays in the file.
  it("says a login written by a newer gxy is not removed, exits 65, and leaves the file as it was", async () => {
    await mkdir(file.dir, { mode: 0o700 });
    await writeFile(
      file.path,
      `hosts:\n  ${HOST}:\n    username: u\n    method: pasted\n    storage: vault\n    obtained_at: "2026-11-06T10:44:00.000Z"\n`,
      { mode: 0o600 },
    );
    const before = await readFile(file.path, "utf8");
    const h = logoutHarness();
    expect(await logoutCommand({}, h.deps)).toBe(65);
    expect(h.cap.err).toContain("written by a newer version of gxy");
    expect(h.cap.err).not.toContain("nothing to remove");
    expect(await readFile(file.path, "utf8")).toBe(before);
  });

  it("deletes a file-stored token too, leaving no copy in hosts.yml", async () => {
    keyring.mode = "throw";
    await store.put(fakeLogin(), { insecureStorage: true });
    expect(await readFile(file.path, "utf8")).toContain(fakeLogin().token);
    expect(await logoutCommand({}, logoutHarness().deps)).toBe(0);
    expect(await readFile(file.path, "utf8")).not.toContain(fakeLogin().token);
  });

  // The record goes either way, and the failure is reported rather than swallowed: a secret that
  // may remain in the keyring is something the person must know, so the exit is not 0.
  it("says the secret may remain, and exits 1, when the keyring will not confirm the delete", async () => {
    await store.put(fakeLogin());
    keyring.mode = "undeletable";
    const h = logoutHarness();
    expect(await logoutCommand({}, h.deps)).toBe(1);
    expect(await file.list()).toEqual([]);
    expect(h.cap.err).toContain("may remain");
    expect(h.cap.err).not.toContain(FAKE_TOKEN);
    // No tick for a removal that did not happen, and the revoke line is kept.
    expect(h.cap.err).not.toContain("✓ Removed");
    expect(h.cap.err).toContain("Could not remove the stored login for dev.galaxy.example.org");
    expect(h.cap.err).toContain("Could not revoke the token on the server");
  });
});
