import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { FileStore } from "../../src/auth/file-store";
import { FileCredentialStore } from "../../src/auth/store";
import { FAKE_TOKEN, HOST, OTHER, fakeLogin, useTempDir } from "./harness";

// The real store over a real `hosts.yml` in a temp directory: the token is kept in the file, next
// to the rest of the login, and is left out of what `list` returns.
const tmp = useTempDir("gxy-st-");
const open = () => {
  const file = new FileStore(join(tmp.path, "cfg"));
  return { file, store: new FileCredentialStore(file) };
};

describe("the file credential store", () => {
  it("puts the token in hosts.yml, with storage: file", async () => {
    const { file, store } = open();
    await store.put(fakeLogin());
    const text = await readFile(file.path, "utf8");
    expect(text).toContain(FAKE_TOKEN);
    expect(text).toMatch(/storage: file/);
  });

  it("gets back what was put, and null for a Galaxy it has no login for", async () => {
    const { store } = open();
    await store.put(fakeLogin());
    expect(await store.get(HOST)).toEqual({ ...fakeLogin(), storage: "file" });
    expect(await store.get(OTHER)).toBeNull();
  });

  it("replaces the login for a Galaxy on a second put", async () => {
    const { store } = open();
    await store.put(fakeLogin());
    await store.put(fakeLogin({ token: "fake-token-second-0123456789" }));
    expect((await store.get(HOST))?.token).toBe("fake-token-second-0123456789");
    expect(await store.list()).toHaveLength(1);
  });

  it("deletes a login, and says false the second time", async () => {
    const { store } = open();
    await store.put(fakeLogin());
    expect(await store.delete(HOST)).toBe(true);
    expect(await store.get(HOST)).toBeNull();
    expect(await store.delete(HOST)).toBe(false);
  });

  it("lists the logins without their tokens, sorted by base URL", async () => {
    const { store } = open();
    await store.put(fakeLogin({ baseUrl: OTHER }));
    await store.put(fakeLogin());
    const list = await store.list();
    expect(list.map((l) => l.baseUrl)).toEqual([HOST, OTHER].sort());
    expect(JSON.stringify(list)).not.toContain(FAKE_TOKEN);
    expect(list[0]).toMatchObject({ method: "pasted", storage: "file" });
  });

  it("rejects, and leaves the file alone, when hosts.yml is corrupt", async () => {
    const { file, store } = open();
    await mkdir(file.dir, { recursive: true, mode: 0o700 });
    await writeFile(file.path, "hosts: [not, a, mapping]\n", { mode: 0o600 });
    await expect(store.get(HOST)).rejects.toThrow(/not valid/);
    await expect(store.list()).rejects.toThrow(/not valid/);
    await expect(store.put(fakeLogin())).rejects.toThrow(/not valid/);
    expect(await readFile(file.path, "utf8")).toBe("hosts: [not, a, mapping]\n");
  });
});
