import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { beforeEach, describe, expect, it } from "vitest";
import { logoutCommand } from "../../src/auth/commands/logout";
import { FileStore } from "../../src/auth/file-store";
import { FileCredentialStore } from "../../src/auth/store";
import { FAKE_TOKEN, HOST, fakeLogin, harness, useTempDir } from "./harness";

describe("auth logout", () => {
  // A pasted key may be in use elsewhere, so logout only forgets it and says to manage it in
  // Galaxy's user preferences instead.
  it("deletes a pasted key without contacting the server and says it was not revoked", async () => {
    const h = harness({ stored: [fakeLogin()] });
    expect(await logoutCommand({}, h.deps)).toBe(0);
    expect(h.whoamiCalls).toEqual([]);
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

  it("is a no-op with a note when nothing is stored for the host", async () => {
    const h = harness();
    expect(await logoutCommand({ host: HOST }, h.deps)).toBe(0);
    expect(h.cap.err).toContain("nothing to remove");
  });
});

// The real store over a real `hosts.yml`, because what logout must leave behind (no token in the
// file) is what the real store does, not what a fake store says.
describe("logout over the real store", () => {
  const tmp = useTempDir("gxy-lo-");
  let file: FileStore;
  let store: FileCredentialStore;
  beforeEach(() => {
    file = new FileStore(join(tmp.path, "cfg"));
    store = new FileCredentialStore(file);
  });

  it("deletes the record, leaving no copy of the token in hosts.yml", async () => {
    await store.put(fakeLogin());
    expect(await readFile(file.path, "utf8")).toContain(fakeLogin().token);
    const h = harness();
    h.deps.store = store;
    h.deps.resolveHost = async () => HOST;
    expect(await logoutCommand({}, h.deps)).toBe(0);
    expect(await file.list()).toEqual([]);
    expect(await readFile(file.path, "utf8")).not.toContain(fakeLogin().token);
    expect(h.cap.err).toContain("Removed the stored API key for dev.galaxy.example.org");
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
    const h = harness();
    h.deps.store = store;
    h.deps.resolveHost = async () => HOST;
    expect(await logoutCommand({}, h.deps)).toBe(65);
    expect(h.cap.err).toContain("written by a newer version of gxy");
    expect(h.cap.err).not.toContain("nothing to remove");
    expect(await readFile(file.path, "utf8")).toBe(before);
  });
});
