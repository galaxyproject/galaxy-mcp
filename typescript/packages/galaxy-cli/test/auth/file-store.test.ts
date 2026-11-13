import {
  chmod,
  mkdir,
  readdir,
  readFile,
  stat,
  symlink,
  utimes,
  writeFile,
} from "node:fs/promises";
import { join } from "node:path";
import { parse as parseYaml } from "yaml";
import { describe, expect, it } from "vitest";
import { FileStore, type HostRecord } from "../../src/auth/file-store";
import type { BaseUrl } from "../../src/auth/types";
import { HOST, OTHER, fakeLogin, useTempDir } from "./harness";

// These run the real `FileStore` on a real directory. What it guards (modes, links, owners, the
// rename into place) belongs to the file system, so an in-memory stand-in could only agree with
// the code rather than check it.
const tmp = useTempDir("gxy-fs-");
/** A config directory that does not exist yet, so a first write has to create it, and set its mode. */
const dirOf = () => join(tmp.path, "cfg");

/** A file-stored record built from `fakeLogin`, so its token is in the file; `over` changes fields. */
function rec(over: Partial<HostRecord> = {}): HostRecord {
  const { token, ...rest } = fakeLogin();
  return { ...rest, storage: "file", token, ...over };
}
const mode = async (p: string) => (await stat(p)).mode & 0o777;
/**
 * What a refusal looks like: an `AuthError` of kind `store`, so a command prints its message as it
 * is and exits with the store's code, not "Unexpected error" (a bug in gxy), and the message says
 * what the person can do.
 */
const refused = (message: RegExp) => ({
  name: "AuthError",
  kind: "store",
  message: expect.stringMatching(message) as unknown,
});

describe("reading and writing", () => {
  it("treats a missing or empty file as no logins", async () => {
    const s = new FileStore(dirOf());
    expect(await s.list()).toEqual([]);
    await mkdir(s.dir);
    await writeFile(s.path, "");
    expect(await s.list()).toEqual([]);
  });

  it("round-trips a record, with its optional fields", async () => {
    const s = new FileStore(dirOf());
    const r = rec({ email: "a@example.org", expiresAt: "2030-01-01T00:00:00.000Z" });
    await s.put(r);
    expect(await s.get(HOST)).toEqual(r);
  });

  it("round-trips a record with no expiry", async () => {
    const s = new FileStore(dirOf());
    await s.put(rec({ expiresAt: null }));
    expect((await s.get(HOST))?.expiresAt).toBeNull();
  });

  it("omits the token from the file for keyring storage", async () => {
    const s = new FileStore(dirOf());
    await s.put({ ...rec(), storage: "keyring", token: undefined });
    expect(await readFile(s.path, "utf8")).not.toMatch(/token/);
  });

  it("deletes, and lists sorted by URL", async () => {
    const s = new FileStore(dirOf());
    // OTHER goes in first and sorts last, so the order read back is not the order written.
    await s.put(rec({ baseUrl: OTHER }));
    await s.put(rec());
    expect((await s.list()).map((r) => r.baseUrl)).toEqual([HOST, OTHER]);
    expect(await s.delete(OTHER)).toBe(true);
    expect(await s.delete(OTHER)).toBe(false);
    expect((await s.list()).map((r) => r.baseUrl)).toEqual([HOST]);
  });
});

describe("permissions", () => {
  it("creates a 0700 directory and a 0600 file, and leaves no temp file", async () => {
    const s = new FileStore(dirOf());
    await s.put(rec());
    // The second put replaces a file that exists, which is where a temp file surviving the
    // rename would show.
    await s.put(rec({ username: "bob" }));
    expect(await mode(s.dir)).toBe(0o700);
    expect(await mode(s.path)).toBe(0o600);
    expect(await readdir(s.dir)).toEqual(["hosts.yml"]);
  });

  // A umask of 000 strips nothing, so a directory made without an explicit 0700 would come out
  // 0777 here. Any wider umask would hide the difference.
  it("still creates the directory 0700 under a umask of 000", async () => {
    const s = new FileStore(dirOf());
    const old = process.umask(0o000);
    try {
      await s.put(rec());
    } finally {
      process.umask(old);
    }
    expect(await mode(s.dir)).toBe(0o700);
  });

  // Only a umask that strips bits from 0600 can show a missing `fchmod`: under 000 or 022 the
  // creation mode alone already gives 0600. Under 0277 the new file would be 0400 without it.
  // The directory is made first because 0277 would leave one the store creates 0500, and the temp
  // file could not then be written.
  it("still writes the file 0600 under a restrictive umask", async () => {
    const s = new FileStore(dirOf());
    await mkdir(s.dir, { mode: 0o700 });
    const old = process.umask(0o277);
    try {
      await s.put(rec());
    } finally {
      process.umask(old);
    }
    expect(await mode(s.path)).toBe(0o600);
  });

  // A looser mode is repaired with a warning, where a symlink or another owner (below) is refused:
  // the file is still ours, only its mode is wrong. Both the file and the directory are fixed on
  // read and on write. `warnings` records what would go to stderr.
  it("repairs a looser file and directory on read, and a looser directory on write, warning each time", async () => {
    const warnings: string[] = [];
    const s = new FileStore(dirOf(), undefined, (m) => warnings.push(m));
    await s.put(rec());
    await chmod(s.path, 0o644);
    await chmod(s.dir, 0o755);
    await s.list();
    expect(await mode(s.path)).toBe(0o600);
    expect(await mode(s.dir)).toBe(0o700);
    expect(warnings.some((w) => /readable by others.*600/.test(w))).toBe(true);
    expect(warnings.some((w) => /accessible to others.*700/.test(w))).toBe(true);
    warnings.length = 0;
    await chmod(s.dir, 0o755);
    await s.put(rec());
    expect(await mode(s.dir)).toBe(0o700);
    expect(warnings.some((w) => /accessible to others.*700/.test(w))).toBe(true);
  });

  // An unparseable file can still hold a token, so its mode is narrowed before the parse is
  // attempted and stays narrow when the parse fails.
  it("repairs a loose file mode before parsing, even when the content then fails to parse", async () => {
    await mkdir(dirOf(), { mode: 0o700 });
    const warnings: string[] = [];
    const s = new FileStore(dirOf(), undefined, (m) => warnings.push(m));
    await writeFile(s.path, "hosts: [unclosed");
    await chmod(s.path, 0o644);
    await expect(s.list()).rejects.toMatchObject(refused(/not valid/));
    expect(await mode(s.path)).toBe(0o600);
    expect(warnings.join()).toMatch(/changed it to 600/);
  });
});

describe("a hosts.yml others can write", () => {
  // Narrowing the mode would not undo an edit someone else may already have made (a login with
  // their token for a real Galaxy URL), so the file is refused, read or write, and left alone.
  it.each([0o666, 0o620, 0o602])(
    "refuses mode %o, and changes neither bytes nor mode",
    async (m) => {
      const s = new FileStore(dirOf());
      await s.put(rec());
      const before = await readFile(s.path, "utf8");
      await chmod(s.path, m);
      await expect(s.list()).rejects.toMatchObject(refused(/writable by others/));
      await expect(s.put(rec({ baseUrl: OTHER }))).rejects.toMatchObject(
        refused(/writable by others/),
      );
      await expect(s.delete(HOST)).rejects.toMatchObject(refused(/writable by others/));
      expect(await mode(s.path)).toBe(m);
      expect(await readFile(s.path, "utf8")).toBe(before);
    },
  );

  it("still repairs a file that others can only read", async () => {
    const s = new FileStore(dirOf());
    await s.put(rec());
    await chmod(s.path, 0o640);
    expect(await s.list()).toHaveLength(1);
    expect(await mode(s.path)).toBe(0o600);
  });
});

describe("refusing what is not a plain file or directory of ours", () => {
  // The target sits outside the config directory and must come out unchanged: a put that followed
  // the link would have written the token to wherever it points. The directory listing then
  // shows that no temp file was left behind either.
  it("refuses a hosts.yml that is a symlink, reading or writing, and does not touch the target", async () => {
    const s = new FileStore(dirOf());
    await mkdir(s.dir, { mode: 0o700 });
    const target = join(tmp.path, "elsewhere.yml");
    await writeFile(target, "hosts: {}\n");
    await symlink(target, s.path);
    await expect(s.list()).rejects.toMatchObject(refused(/is a symbolic link/));
    await expect(s.put(rec())).rejects.toMatchObject(refused(/is a symbolic link/));
    expect(await readFile(target, "utf8")).toBe("hosts: {}\n");
    expect(await readdir(s.dir)).toEqual(["hosts.yml"]);
  });

  // `real` must stay empty: a write through the linked directory would land a token in it.
  it("refuses a config directory that is a symlink", async () => {
    const real = join(tmp.path, "real");
    await mkdir(real, { mode: 0o700 });
    const link = join(tmp.path, "link");
    await symlink(real, link);
    const s = new FileStore(link);
    await expect(s.list()).rejects.toMatchObject(refused(/is a symbolic link/));
    await expect(s.put(rec())).rejects.toMatchObject(refused(/is a symbolic link/));
    expect(await readdir(real)).toEqual([]);
  });

  it("refuses a directory or file owned by someone else", async () => {
    // Changing a file's owner needs root, so the test changes the uid the store expects instead:
    // the files are ours, and the store is told that someone else should own them.
    const mine = process.getuid!();
    const s = new FileStore(dirOf(), undefined, undefined, mine + 1);
    await mkdir(s.dir, { mode: 0o700 });
    await expect(s.list()).rejects.toMatchObject(refused(/belongs to another user/));
    await expect(s.put(rec())).rejects.toMatchObject(refused(/belongs to another user/));
    // The same directory is fine for its real owner, and a file of someone else is refused on its own.
    await new FileStore(s.dir, undefined, undefined, mine).put(rec());
    await expect(new FileStore(s.dir, undefined, undefined, mine + 1).list()).rejects.toThrow(
      /belongs to another user/,
    );
  });

  it("refuses a hosts.yml that is a directory", async () => {
    const s = new FileStore(dirOf());
    await mkdir(s.path, { recursive: true });
    await expect(s.list()).rejects.toMatchObject(refused(/not a regular file/));
  });

  // `process.getuid` does not exist on Windows, so the store takes `undefined` there and cannot
  // compare owners. The check is skipped, not failed, or no login could be stored at all.
  it("skips the ownership check when there is no uid to compare (Windows)", async () => {
    const s = new FileStore(dirOf(), undefined, undefined, undefined);
    await s.put(rec());
    expect(await s.list()).toHaveLength(1);
  });
});

describe("a corrupt hosts.yml", () => {
  // Every variant below puts a canary token in a field the YAML parser might quote back in its
  // error, so a message that repeated the file's text would repeat the token. Each test therefore
  // asserts what the message names (the file) and, separately, that the canary is absent.
  const SECRET = "FAKE-SECRET-TOKEN-123";
  const entry = (over: Record<string, string> = {}) => {
    const fields: Record<string, string> = {
      username: "aioue",
      method: "pasted",
      scopes: "[]",
      expires_at: "null",
      obtained_at: '"2026-11-06T10:44:00.000Z"',
      storage: "file",
      token: SECRET,
      ...over,
    };
    return `hosts:\n  ${HOST}:\n${Object.entries(fields)
      .map(([k, v]) => `    ${k}: ${v}\n`)
      .join("")}`;
  };

  /** Writes `text` as the hosts file; returns the store and the message of the error `list` raises. */
  async function readError(text: string): Promise<{ s: FileStore; message: string }> {
    await mkdir(dirOf(), { recursive: true });
    const s = new FileStore(dirOf());
    await writeFile(s.path, text, { mode: 0o600 });
    const e = await s.list().then(
      () => undefined,
      (x: unknown) => x as Error,
    );
    // A file that was accepted fails here, so a case cannot pass by not being corrupt after all.
    if (!e) throw new Error("expected the corrupt file to be refused");
    return { s, message: e.message };
  }

  it("names the file and the line and column of a syntax error, and nothing from its content", async () => {
    const { s, message } = await readError(
      `hosts:\n  https://a.example:\n    token: ${SECRET} : bad\n    username: [unclosed\n`,
    );
    expect(message).toContain(s.path);
    expect(message).toMatch(/line \d+, column \d+/);
    expect(message).not.toContain(SECRET);
    expect(message).not.toContain("unclosed");
  });

  it.each([
    ["the top level is a list", `- ${SECRET}\n`],
    ["the top level is text", `${SECRET}\n`],
    ["hosts is not a mapping", `hosts: ${SECRET}\n`],
    ["an entry is not a mapping", `hosts:\n  ${HOST}: ${SECRET}\n`],
    ["scopes is not a list", entry({ scopes: "5" })],
    ["scopes holds a non-string", entry({ scopes: "[1]" })],
    ["expires_at is not a string", entry({ expires_at: "5" })],
    ["a required field is missing", entry({ username: '""' })],
    ["a mapping has other keys and no hosts key", `Hosts:\n  ${HOST}: {}\nother: ${SECRET}\n`],
    ["a file-stored entry has no token", entry({ token: '""' })],
  ])("names only the file when %s", async (_what, text) => {
    const { s, message } = await readError(text);
    expect(message).toContain(s.path);
    expect(message).toMatch(/not valid/);
    expect(message).not.toContain(SECRET);
  });

  // A put that treated an unreadable file as empty would replace every other host's login with
  // one entry. The message says "it has not been modified", so the bytes are compared.
  it("does not overwrite it on a put, and a put names it too", async () => {
    const text = "hosts: [unclosed\n  : :";
    const { s } = await readError(text);
    await expect(s.put(rec())).rejects.toThrow(s.path);
    expect(await readFile(s.path, "utf8")).toBe(text);
  });

  // The file says nothing we can place, so reading it as empty would let the next put replace
  // it. Only an empty or null document, or an empty `hosts`, is empty.
  it.each([["other: 1\n"], ["host:\n  a: b\n"], ["version: 2\nentries: {}\n"]])(
    "does not read a mapping with no hosts key as empty, and a put leaves it alone: %j",
    async (text) => {
      const { s } = await readError(text);
      await expect(s.put(rec())).rejects.toMatchObject(refused(/no `hosts` key/));
      expect(await readFile(s.path, "utf8")).toBe(text);
    },
  );

  it.each([["{}\n"], ["hosts:\n"], ["hosts: {}\n"], ["# nothing yet\n"], ["null\n"]])(
    "reads %j as no logins",
    async (text) => {
      await mkdir(dirOf(), { recursive: true });
      const s = new FileStore(dirOf());
      await writeFile(s.path, text, { mode: 0o600 });
      expect(await s.list()).toEqual([]);
      await s.put(rec());
      expect(await s.list()).toHaveLength(1);
    },
  );
});

// A later (or earlier) gxy can write entries this one cannot read. One of them must not lock out
// the logins this build does understand, and no write may lose it.
describe("entries this build does not recognise", () => {
  const SECRET = "FAKE-SECRET-TOKEN-123";
  const FUTURE: BaseUrl = "https://future.example" as BaseUrl;
  /** A hosts file with one login we understand (HOST, file storage) and `extra` yaml after it. */
  const fileWith = (extra: string) =>
    `hosts:\n  ${HOST}:\n    username: aioue\n    method: pasted\n    scopes: []\n    expires_at: null\n    obtained_at: "2026-11-06T10:44:00.000Z"\n    storage: file\n    token: ${SECRET}\n${extra}`;
  const futureEntry = (storage = "vault") =>
    `  ${FUTURE}:\n    username: bob\n    method: pasted\n    storage: ${storage}\n    handle: opaque-123\n`;

  async function open(text: string): Promise<{ s: FileStore; warnings: string[] }> {
    await mkdir(dirOf(), { recursive: true, mode: 0o700 });
    const warnings: string[] = [];
    const s = new FileStore(dirOf(), undefined, (m) => warnings.push(m));
    await writeFile(s.path, text, { mode: 0o600 });
    return { s, warnings };
  }

  it("skips an entry with an unknown storage or method, warns once, and reads the rest", async () => {
    const { s, warnings } = await open(
      fileWith(
        futureEntry("vault") +
          futureEntry()
            .replace(FUTURE, "https://m.example")
            .replace("vault", "file")
            .replace("pasted", "passkey"),
      ),
    );
    expect((await s.list()).map((r) => r.baseUrl)).toEqual([HOST]);
    expect((await s.get(HOST))?.token).toBe(SECRET);
    expect(await s.list()).toHaveLength(1);
    expect(warnings).toHaveLength(2);
    expect(warnings[0]).toContain(s.path);
    expect(warnings[0]).toContain("unknown storage");
    expect(warnings[1]).toContain("unknown method");
    // Neither the value nor anything else in the entry is echoed.
    expect(warnings.join()).not.toMatch(/vault|opaque|passkey/);
  });

  it("refuses to use the skipped host itself, rather than reporting it as not logged in", async () => {
    const { s } = await open(fileWith(futureEntry()));
    await expect(s.get(FUTURE)).rejects.toMatchObject(refused(/written by a newer version of gxy/));
  });

  // The write must carry the entry out unchanged, with its unknown fields, and also unknown
  // fields on an entry we do read and unknown top-level keys.
  it("keeps what it skips when it writes: unknown entries, unknown fields and top-level keys", async () => {
    const text =
      fileWith(futureEntry()).replace(
        "    storage: file\n",
        "    storage: file\n    flavour: mint\n",
      ) + "schema: 7\n";
    const { s } = await open(text);
    await s.put(rec({ baseUrl: OTHER }));
    const written = parseYaml(await readFile(s.path, "utf8")) as {
      hosts: Record<string, Record<string, unknown>>;
      schema: number;
    };
    expect(Object.keys(written.hosts)).toEqual([HOST, FUTURE, OTHER]);
    expect(written.hosts[FUTURE]).toEqual({
      username: "bob",
      method: "pasted",
      storage: "vault",
      handle: "opaque-123",
    });
    expect(written.hosts[HOST]?.flavour).toBe("mint");
    expect(written.schema).toBe(7);
    // And deleting a host we do understand keeps the rest as well.
    expect(await s.delete(OTHER)).toBe(true);
    expect(await readFile(s.path, "utf8")).toContain("handle: opaque-123");
  });

  // Replacing it would discard a login (and maybe a keyring secret) that cannot be read.
  it("refuses to replace or delete the skipped host, and leaves the file as it was", async () => {
    const { s } = await open(fileWith(futureEntry()));
    const before = await readFile(s.path, "utf8");
    await expect(s.put(rec({ baseUrl: FUTURE }))).rejects.toMatchObject(
      refused(/written by a newer version of gxy/),
    );
    await expect(s.delete(FUTURE)).rejects.toMatchObject(
      refused(/written by a newer version of gxy/),
    );
    expect(await readFile(s.path, "utf8")).toBe(before);
  });

  // `unknown storage` wins over other faults in the same entry: we cannot judge what another
  // version wrote by this one's field rules.
  it("skips an entry of another version even when it lacks fields this one requires", async () => {
    const { s } = await open(fileWith(`  ${FUTURE}:\n    storage: vault\n`));
    expect((await s.list()).map((r) => r.baseUrl)).toEqual([HOST]);
  });

  // Keys that are not normalised URLs were not written by this build and are never looked up. An
  // `http://` key to a non-loopback host in particular must not be trusted with a token.
  it.each(["http://evil.example", "HTTPS://Host.Example", "not a url", "https://h.example/api"])(
    "skips a key that is not a normalised Galaxy URL: %s",
    async (key) => {
      const { s, warnings } = await open(
        fileWith(
          `  ${JSON.stringify(key)}:\n    username: x\n    method: pasted\n    storage: file\n    token: ${SECRET}\n    obtained_at: "2026-11-06T10:44:00.000Z"\n`,
        ),
      );
      expect((await s.list()).map((r) => r.baseUrl)).toEqual([HOST]);
      expect(warnings.join()).toMatch(/not a normalised Galaxy URL/);
      expect(warnings.join()).not.toContain(SECRET);
    },
  );

  // File text goes into the terminal, so control characters in a key do not.
  it("strips control characters from a file-supplied key in a warning", async () => {
    const { s, warnings } = await open(
      fileWith(`  "https://x.example\\u001b[2J\\u0007":\n    storage: file\n`),
    );
    await s.list();
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain("x.example[2J");
    expect(warnings[0]).not.toMatch(/[\u0000-\u001f]/);
  });

  // `out["__proto__"] = ...` on a plain object sets the prototype and loses the entry.
  it("treats a host key of __proto__ as a key: it is skipped, kept, and does not eat others", async () => {
    const { s } = await open(fileWith("  __proto__:\n    username: x\n    storage: file\n"));
    expect((await s.list()).map((r) => r.baseUrl)).toEqual([HOST]);
    await s.put(rec({ baseUrl: OTHER }));
    const text = await readFile(s.path, "utf8");
    expect(text).toContain("__proto__:");
    expect((await s.list()).map((r) => r.baseUrl)).toEqual([HOST, OTHER].sort());
  });
});

describe("stale temp files", () => {
  // A write killed before its rename leaves a temp file that may hold a token. Age is the only
  // thing that tells a leftover from a write still running in another process, so the fresh one
  // stays; `notes.tmp` is old but not ours, so it stays too. The ages are set with `utimes`
  // relative to now, not by waiting.
  it("removes .hosts.*.tmp files older than an hour on a write, and only those", async () => {
    await mkdir(dirOf(), { mode: 0o700 });
    const old = join(dirOf(), ".hosts.123.abc.tmp");
    const fresh = join(dirOf(), ".hosts.456.def.tmp");
    const other = join(dirOf(), "notes.tmp");
    for (const f of [old, fresh, other]) await writeFile(f, "x");
    const twoHoursAgo = new Date(Date.now() - 2 * 3600_000);
    await utimes(old, twoHoursAgo, twoHoursAgo);
    await utimes(other, twoHoursAgo, twoHoursAgo);
    await new FileStore(dirOf()).put(rec());
    expect((await readdir(dirOf())).sort()).toEqual([
      ".hosts.456.def.tmp",
      "hosts.yml",
      "notes.tmp",
    ]);
  });
});
