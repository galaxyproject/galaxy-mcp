// A write that fails midway, and the rule that the temp file is never a pre-planted symlink. Both need the
// filesystem and the temp-file name under test control, so `node:fs/promises` and `randomBytes`
// are wrapped here (passing through unless a test arms them). This is a file of its own because
// `vi.mock` applies to the whole file, and the other store tests need the real file system.
import { mkdir, readdir, readFile, symlink, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type * as Crypto from "node:crypto";
import type * as FsPromises from "node:fs/promises";
import { beforeEach, describe, expect, it, vi } from "vitest";

/**
 * The switches the wrappers below read, set by a test before the call it wants to fail. `failWrite`
 * and `failRename` make that step throw; `fixedRandom` makes the temp file's name predictable.
 * Hoisted because `vi.mock` factories run before the file's own statements.
 */
const arm = vi.hoisted(() => ({
  failWrite: false,
  failRename: false,
  fixedRandom: undefined as number[] | undefined,
}));

// With `fixedRandom` set the temp name is `.hosts.<pid>.<those bytes in hex>.tmp`, so a test can
// plant something at exactly that path. Unset, it is the real `randomBytes`.
vi.mock("node:crypto", async (orig) => {
  const actual = await orig<typeof Crypto>();
  return {
    ...actual,
    randomBytes: (n: number) =>
      arm.fixedRandom ? Buffer.from(arm.fixedRandom) : actual.randomBytes(n),
  };
});

// `open` returns the real handle, wrapped so that `writeFile` on a `.hosts.` temp file throws
// "disk full" once armed. The failure has to come after the file exists, or the cleanup that
// removes a half-written temp file (which would hold the token) would never run. `rename` throws
// "rename refused" when armed, which fails the last step with the temp file complete on disk.
vi.mock("node:fs/promises", async (orig) => {
  const actual = await orig<typeof FsPromises>();
  return {
    ...actual,
    async open(path: string, flags?: unknown, mode?: unknown) {
      const fh = await (actual.open as (...a: unknown[]) => Promise<FsPromises.FileHandle>)(
        path,
        flags,
        mode,
      );
      if (!arm.failWrite || !path.includes(".hosts.")) return fh;
      return new Proxy(fh, {
        get(target, prop) {
          if (prop === "writeFile")
            return async () => {
              throw new Error("disk full");
            };
          const v = Reflect.get(target, prop) as unknown;
          return typeof v === "function" ? (v as (...a: unknown[]) => unknown).bind(target) : v;
        },
      });
    },
    async rename(from: string, to: string) {
      if (arm.failRename) throw new Error("rename refused");
      return actual.rename(from, to);
    },
  };
});

const { FileStore } = await import("../../src/auth/file-store");
const { FAKE_TOKEN, fakeLogin, useTempDir } = await import("./harness");

const tmp = useTempDir("gxy-fsw-");
// Disarmed before every test, so a failure one test switched on cannot carry into the next.
beforeEach(() => {
  arm.failWrite = false;
  arm.failRename = false;
  arm.fixedRandom = undefined;
});

const rec = () => {
  const { token, ...rest } = fakeLogin();
  return { ...rest, storage: "file" as const, token };
};
/** The hosts file before the failing put: valid, and different from anything `put` would write. */
const prior = "hosts: {}\n";

// The write and the rename have separate cleanup in the store, so each is failed in turn.
describe.each([
  [
    "writing the temp file",
    () => {
      arm.failWrite = true;
    },
    /disk full/,
  ],
  [
    "renaming it into place",
    () => {
      arm.failRename = true;
    },
    /rename refused/,
  ],
] as const)("a failure while %s", (_when, breakIt, why) => {
  /** Puts into a directory that already holds `prior`, with the failure armed; returns the store. */
  async function failedPut() {
    const dir = join(tmp.path, "cfg");
    await mkdir(dir, { mode: 0o700 });
    const s = new FileStore(dir);
    await writeFile(s.path, prior, { mode: 0o600 });
    breakIt();
    await expect(s.put(rec())).rejects.toThrow(why);
    return s;
  }

  it("propagates the error and leaves the existing hosts.yml as it was", async () => {
    const s = await failedPut();
    expect(await readFile(s.path, "utf8")).toBe(prior);
  });

  // A temp file left by the failure would be a second, unlisted copy of the token.
  it("leaves no temp file, which would hold the token", async () => {
    const s = await failedPut();
    expect(await readdir(s.dir)).toEqual(["hosts.yml"]);
  });
});

describe("the temp file", () => {
  // Someone who can write to the directory could plant a symlink at the name the temp file is about
  // to take. The name is made predictable here to do exactly that, pointing at `victim`. The expected
  // `EEXIST` is the exclusive create refusing an existing path, which a plain open would instead
  // have followed, writing the token into `victim`.
  it("is never a pre-planted symlink: the write fails rather than follow it", async () => {
    const dir = join(tmp.path, "cfg");
    await mkdir(dir, { mode: 0o700 });
    arm.fixedRandom = [1, 2, 3, 4, 5, 6];
    const victim = join(tmp.path, "victim");
    await writeFile(victim, "untouched");
    await symlink(victim, join(dir, `.hosts.${process.pid}.010203040506.tmp`));
    const s = new FileStore(dir);
    await expect(s.put(rec())).rejects.toThrow(/EEXIST/);
    expect(await readFile(victim, "utf8")).toBe("untouched");
    expect(await readFile(victim, "utf8")).not.toContain(FAKE_TOKEN);
  });
});
