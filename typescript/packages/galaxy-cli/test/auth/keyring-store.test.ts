import { spawn } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { createServer, type Socket } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { describe, expect, it, vi } from "vitest";
import {
  childEnv,
  KeyringStore,
  KeyringUnavailableError,
  nativeBackend,
} from "../../src/auth/keyring-store";
import { HOST as baseUrl, FakeKeyring, useTempDir } from "./harness";

describe("KeyringStore", () => {
  // The key is spelled out here rather than built with `keyringService`: a change to its format
  // would orphan every token already stored under the old one, and this is where that shows.
  it("keys entries by gxy:<baseUrl> and username", async () => {
    const kr = new FakeKeyring();
    const s = new KeyringStore(kr, 50);
    await s.set(baseUrl, "alice", "FAKE");
    expect([...kr.data.keys()]).toEqual([`gxy:${baseUrl}|alice`]);
    expect(await s.get(baseUrl, "alice")).toBe("FAKE");
    expect(await s.delete(baseUrl, "alice")).toBe(true);
  });
  // `FakeKeyring` in "hang" mode never settles, so only the store's own timer can end the call.
  it("times out a hung backend", async () => {
    const kr = new FakeKeyring();
    kr.mode = "hang";
    await expect(new KeyringStore(kr, 20).get(baseUrl, "alice")).rejects.toThrow(
      KeyringUnavailableError,
    );
  });
  it("wraps backend errors", async () => {
    const kr = new FakeKeyring();
    kr.mode = "throw";
    await expect(new KeyringStore(kr, 50).set(baseUrl, "a", "x")).rejects.toThrow(
      /system keyring failed/,
    );
  });
});

// A person at the terminal can answer a keyring unlock prompt, which blocks the call until they
// type; 5 s would kill it. The wait is longer then, and the person is told once it is a wait.
describe("KeyringStore with a person at the terminal", () => {
  const tmp = useTempDir("gxy-ki-");
  /** A child that answers a `get` with "late" after `ms`. */
  const slowChild = async (ms: number) => {
    const p = join(tmp.path, "slow.mjs");
    await writeFile(
      p,
      `for await (const _ of process.stdin);
       setTimeout(() => process.stdout.write(JSON.stringify({ ok: true, value: "late" })), ${ms});`,
    );
    return nativeBackend({ script: p, env: { PATH: process.env.PATH } });
  };

  it("waits past the usual timeout for an unlock prompt, and says it is waiting, once", async () => {
    const notes: string[] = [];
    const s = new KeyringStore(await slowChild(600), 150, {
      interactive: true,
      interactiveTimeoutMs: 8000,
      noticeAfterMs: 100,
      notify: (m) => notes.push(m),
    });
    expect(await s.get(baseUrl, "alice")).toBe("late");
    expect(await s.get(baseUrl, "alice")).toBe("late");
    expect(notes).toHaveLength(1);
    expect(notes[0]).toMatch(/waiting for the system keyring.*unlock/i);
  });

  it("keeps the short timeout when nobody can answer, and says nothing", async () => {
    const notes: string[] = [];
    const s = new KeyringStore(await slowChild(2000), 150, {
      noticeAfterMs: 20,
      notify: (m) => notes.push(m),
    });
    await expect(s.get(baseUrl, "alice")).rejects.toThrow(/did not answer a get within 150 ms/);
    expect(notes).toEqual([]);
  });

  it("says nothing about a keyring that answers at once", async () => {
    const notes: string[] = [];
    const s = new KeyringStore(new FakeKeyring(), 50, {
      interactive: true,
      noticeAfterMs: 30,
      notify: (m) => notes.push(m),
    });
    await s.set(baseUrl, "alice", "FAKE");
    await new Promise((r) => setTimeout(r, 60));
    expect(notes).toEqual([]);
  });

  it("gives up after the interactive timeout when nothing is typed", async () => {
    const s = new KeyringStore(new FakeKeyring(), 5000, {
      interactive: true,
      interactiveTimeoutMs: 40,
    });
    const kr = new FakeKeyring();
    kr.mode = "hang";
    await expect(
      new KeyringStore(kr, 5000, { interactive: true, interactiveTimeoutMs: 40 }).get(baseUrl, "a"),
    ).rejects.toThrow(/within 40 ms/);
    expect(await s.get(baseUrl, "a")).toBeNull();
  });
});

// One command, one wait: a keyring that did not answer will not answer the next call either, and
// each further wait is another timeout stacked on a login (write, then cleanup) or a logout.
describe("KeyringStore after a timeout", () => {
  it("fails the rest of its calls at once without asking the backend again", async () => {
    const kr = new FakeKeyring();
    kr.mode = "hang";
    const s = new KeyringStore(kr, 30);
    await expect(s.get(baseUrl, "a")).rejects.toThrow(/within 30 ms/);
    await expect(s.delete(baseUrl, "a")).rejects.toThrow(/did not answer earlier/);
    await expect(s.set(baseUrl, "a", "x")).rejects.toThrow(KeyringUnavailableError);
    expect(kr.calls).toEqual(["get"]);
  });

  it("is not wedged by an error that is not a timeout", async () => {
    const kr = new FakeKeyring();
    kr.mode = "throw";
    const s = new KeyringStore(kr, 30);
    await expect(s.get(baseUrl, "a")).rejects.toThrow(/failed a read/);
    await expect(s.get(baseUrl, "a")).rejects.toThrow(/failed a read/);
    expect(kr.calls).toEqual(["get", "get"]);
  });

  // A backend that enforces its own timeout and kills what it started is not raced: the store's
  // timer would fire first and a caller that exits on the rejection would orphan the child.
  it("does not race a backend that times itself", async () => {
    const slow = {
      selfTimed: true,
      get: () =>
        new Promise<string | null>((r) =>
          setTimeout(() => {
            r("x");
          }, 80),
        ),
      set: () => Promise.resolve(),
      delete: () => Promise.resolve(true),
    };
    expect(await new KeyringStore(slow, 20).get(baseUrl, "a")).toBe("x");
  });
});

// A backend can report a delete that did not happen (a locked collection, a different item
// matched). `FakeKeyring` in "undeletable" mode does exactly that: it answers true and keeps the
// secret. A token that quietly stays after `logout` is the outcome the read-back exists to prevent.
describe("KeyringStore.delete is verified", () => {
  it("reads back after deleting and reports the secret may remain when it is still there", async () => {
    const kr = new FakeKeyring();
    const s = new KeyringStore(kr, 50);
    await s.set(baseUrl, "alice", "FAKE");
    kr.mode = "undeletable";
    await expect(s.delete(baseUrl, "alice")).rejects.toThrow(/may remain/);
  });
  it("reports false when there was nothing to delete, and does not throw", async () => {
    expect(await new KeyringStore(new FakeKeyring(), 50).delete(baseUrl, "alice")).toBe(false);
  });
});

// The real `nativeBackend`, spawning a real Node process, against stub scripts standing in for
// `keyring-child.ts`: the stdin/stdout protocol, the kill on timeout, and what the child is given.
// A real process is the point, because the timeout and the kill are about process behaviour that a
// fake `spawn` could only imitate.
describe("nativeBackend (stub child)", () => {
  const tmp = useTempDir("gxy-kc-");
  /** Writes `body` as an ES module in the test's temp dir and returns its path, to pass as `script`. */
  const stub = async (body: string) => {
    const p = join(tmp.path, "stub.mjs");
    await writeFile(p, body);
    return p;
  };
  // The child's whole environment: only PATH, so no `DBUS_SESSION_BUS_ADDRESS` or `GALAXY_*` from the
  // machine running the tests can reach it.
  const env = { PATH: process.env.PATH };

  it("sends the request as JSON on stdin and reads the JSON reply", async () => {
    const script = await stub(`
      let s = ""; for await (const c of process.stdin) s += c;
      const r = JSON.parse(s);
      process.stdout.write(JSON.stringify({ ok: true, value: r.op === "get" ? r.service + "/" + r.account : true }));`);
    const b = nativeBackend({ script, env, timeoutMs: 5000 });
    expect(await b.get("svc", "acct")).toBe("svc/acct");
    await b.set("svc", "acct", "FAKE");
    expect(await b.delete("svc", "acct")).toBe(true);
  });

  // Argv is visible to other users through `ps`, and a secret in the environment is inherited by
  // anything the child starts, so the secret may travel only on stdin. No `env` is passed, so the
  // child gets the default environment `nativeBackend` builds from this process's: the path
  // production takes. A
  // `GALAXY_API_KEY` is planted in the parent to show that it is not carried across. The stub
  // writes what it saw to a file, since a child's own view of its argv and environment cannot be
  // read from outside.
  it("keeps the secret out of the child's argv and environment, and passes no GALAXY_* variables", async () => {
    const log = join(tmp.path, "log.json");
    const script = await stub(`
      import { writeFileSync } from "node:fs";
      let s = ""; for await (const c of process.stdin) s += c;
      writeFileSync(${JSON.stringify(log)}, JSON.stringify({ argv: process.argv, env: process.env, stdin: s }));
      process.stdout.write(JSON.stringify({ ok: true, value: true }));`);
    const secret = "FAKE-SECRET-VALUE-123";
    const saved = process.env.GALAXY_API_KEY;
    process.env.GALAXY_API_KEY = "FAKE-API-KEY-456";
    try {
      await nativeBackend({ script, timeoutMs: 5000 }).set("svc", "acct", secret);
    } finally {
      if (saved === undefined) delete process.env.GALAXY_API_KEY;
      else process.env.GALAXY_API_KEY = saved;
    }
    const seen = JSON.parse(await readFile(log, "utf8")) as {
      argv: string[];
      env: Record<string, string>;
      stdin: string;
    };
    expect(JSON.stringify(seen.argv)).not.toContain(secret);
    expect(JSON.stringify(seen.env)).not.toContain(secret);
    expect(JSON.stringify(seen.env)).not.toContain("FAKE-API-KEY-456");
    expect(Object.keys(seen.env).some((k) => k.startsWith("GALAXY_"))).toBe(false);
    expect(JSON.parse(seen.stdin)).toMatchObject({ op: "set", secret });
  });

  // An allowlist, so a new secret in the parent's environment is dropped by default.
  it("childEnv keeps what the bus needs and drops everything else", () => {
    const e = childEnv({
      HOME: "/h",
      DBUS_SESSION_BUS_ADDRESS: "unix:path=/x",
      GALAXY_API_KEY: "K",
      AWS_SECRET_ACCESS_KEY: "S",
    });
    expect(e).toEqual({ HOME: "/h", DBUS_SESSION_BUS_ADDRESS: "unix:path=/x" });
  });

  it("kills a child that never answers and reports that the keyring did not answer", async () => {
    const pidFile = join(tmp.path, "pid");
    const script = await stub(`
      import { writeFileSync } from "node:fs";
      writeFileSync(${JSON.stringify(pidFile)}, String(process.pid));
      setInterval(() => {}, 1000);`);
    // The store hands its timeout to the native backend, which is the layer that kills the child
    // and gives up, and the message says so. 2500 ms is wide enough for a loaded machine to start Node and write the pid file: a tight timeout raced the child's
    // start and flaked. The test therefore waits for the pid file instead of racing a timer.
    // `outcome` is attached before the wait, so the rejection is handled whenever it arrives, and
    // is then checked once the pid is known.
    const call = new KeyringStore(nativeBackend({ script, env }), 2500).get(baseUrl, "alice");
    const outcome = call.then(
      () => undefined,
      (e: unknown) => e,
    );
    await vi.waitFor(
      async () => {
        expect(Number(await readFile(pidFile, "utf8"))).toBeGreaterThan(0);
      },
      { timeout: 2000, interval: 20 },
    );
    const pid = Number(await readFile(pidFile, "utf8"));
    const failure = await outcome;
    expect(failure).toBeInstanceOf(Error);
    expect((failure as Error).message).toMatch(/did not answer a get within 2500 ms/);
    // Signal 0 only asks whether the process exists. SIGKILL is delivered at once but the process
    // takes a moment to be reaped; a child left running would hold the CLI open.
    await vi.waitFor(
      () => {
        expect(() => process.kill(pid, 0)).toThrow();
      },
      { timeout: 5000 },
    );
  }, 20_000);

  // The bodies are: output that is not JSON, a clean exit that printed nothing (silence), a crash
  // (non-zero exit, nothing printed), JSON whose `ok` is not a boolean, and JSON `null`. Each must
  // read as "no keyring" so that the store falls back to the file.
  it("maps garbage, silence, a crash and malformed replies to unavailable", async () => {
    for (const body of [
      `process.stdout.write("not json")`,
      `process.exit(0)`,
      `process.exit(3)`,
      `process.stdout.write('{"ok":"yes"}')`,
      `process.stdout.write("null")`,
    ]) {
      const script = await stub(body);
      await expect(
        new KeyringStore(nativeBackend({ script, env, timeoutMs: 5000 }), 5000).get(baseUrl, "a"),
      ).rejects.toThrow(KeyringUnavailableError);
    }
  });

  // The child's three error kinds mean different things to the caller: `unavailable` and `failed`
  // both reach it as errors, with the child's own message, while `not-found` is the ordinary
  // "nothing stored" answer and so becomes null.
  it("maps the child's error kinds", async () => {
    const unavailable = await stub(
      `process.stdout.write(JSON.stringify({ ok: false, kind: "unavailable", message: "no bus" }))`,
    );
    await expect(
      new KeyringStore(nativeBackend({ script: unavailable, env }), 5000).get(baseUrl, "a"),
    ).rejects.toThrow(/no bus/);
    const failed = await stub(
      `process.stdout.write(JSON.stringify({ ok: false, kind: "failed", message: "locked" }))`,
    );
    await expect(
      new KeyringStore(nativeBackend({ script: failed, env }), 5000).set(baseUrl, "a", "x"),
    ).rejects.toThrow(/system keyring failed a write: locked/);
    const notFound = await stub(
      `process.stdout.write(JSON.stringify({ ok: false, kind: "not-found", message: "none" }))`,
    );
    expect(await nativeBackend({ script: notFound, env }).get("s", "a")).toBeNull();
  });

  // A helper that has answered but cannot exit (a native thread still blocked on the bus) must not
  // turn its answer into a timeout, which for a `set` would make a stored token look unstored.
  it("takes a complete reply at once from a child that then hangs, and kills it", async () => {
    const pidFile = join(tmp.path, "pid2");
    const script = await stub(`
      import { writeFileSync } from "node:fs";
      writeFileSync(${JSON.stringify(pidFile)}, String(process.pid));
      process.stdout.write(JSON.stringify({ ok: true, value: "the-secret" }));
      setInterval(() => {}, 1000);`);
    const b = nativeBackend({ script, env, timeoutMs: 20_000 });
    expect(await b.get("svc", "acct")).toBe("the-secret");
    const pid = Number(await readFile(pidFile, "utf8"));
    await vi.waitFor(
      () => {
        expect(() => process.kill(pid, 0)).toThrow();
      },
      { timeout: 5000 },
    );
  }, 15_000);

  it("gives up on a child that floods its output, without waiting for the timeout", async () => {
    const script = await stub(`
      process.stdout.write("x".repeat(200_000));
      setInterval(() => {}, 1000);`);
    await expect(
      nativeBackend({ script, env, timeoutMs: 20_000 }).get("svc", "acct"),
    ).rejects.toThrow(/no usable answer/);
  }, 15_000);

  // The re-check after a delete reads "nothing there" as success, so a reply that is JSON with an
  // `ok` but the wrong shape for its operation must never be taken for an answer.
  it.each([
    ["get", `{"ok":true,"value":5}`],
    ["get", `{"ok":true}`],
    ["get", `{"ok":true,"value":true}`],
    ["set", `{"ok":true,"value":"x"}`],
    ["delete", `{"ok":true,"value":"yes"}`],
    ["delete", `{"ok":true}`],
    ["get", `{"ok":false}`],
    ["get", `{"ok":false,"kind":"weird","message":"m"}`],
    ["get", `{"ok":false,"kind":"failed"}`],
  ] as const)("treats %s with the reply %s as no usable answer", async (op, reply) => {
    const script = await stub(`process.stdout.write(${JSON.stringify(reply)})`);
    const b = nativeBackend({ script, env, timeoutMs: 5000 });
    const run = () =>
      op === "get" ? b.get("s", "a") : op === "set" ? b.set("s", "a", "x") : b.delete("s", "a");
    await expect(run()).rejects.toThrow(/no usable answer/);
  });

  it("reports a child that cannot start", async () => {
    await expect(
      new KeyringStore(
        nativeBackend({ script: join(tmp.path, "missing.mjs"), env, timeoutMs: 5000 }),
        5000,
      ).get(baseUrl, "a"),
    ).rejects.toThrow(KeyringUnavailableError);
  });
});

// The real child script, with `@napi-rs/keyring` replaced by a stub module next to a copy of it: what
// the child asks the native module for, without any chance of reaching a real keyring. The copy
// sits in its own temp directory so that Node resolves `@napi-rs/keyring` to the stub in the
// `node_modules` beside it, and a `package.json` there makes the `.ts` copy load as an ES module.
describe("keyring-child (stub native module)", () => {
  const tmp = useTempDir("gxy-kn-");
  const childSource = join(__dirname, "../../src/auth/keyring-child.ts");

  /** A copy of the child in a temp dir; `stubModule` is the body of the fake `@napi-rs/keyring`, or undefined for none. */
  async function childWith(stubModule: string | undefined) {
    const script = join(tmp.path, "keyring-child.ts");
    await writeFile(script, await readFile(childSource, "utf8"));
    await writeFile(join(tmp.path, "package.json"), '{"type":"module"}');
    if (stubModule !== undefined) {
      const mod = join(tmp.path, "node_modules", "@napi-rs", "keyring");
      await mkdir(mod, { recursive: true });
      await writeFile(
        join(mod, "package.json"),
        '{"name":"@napi-rs/keyring","type":"module","main":"index.js"}',
      );
      await writeFile(join(mod, "index.js"), stubModule);
    }
    return nativeBackend({
      script,
      env: { PATH: process.env.PATH, STUB_LOG: join(tmp.path, "calls.log") },
      timeoutMs: 10_000,
    });
  }
  /**
   * A stand-in for `@napi-rs/keyring`. Every `AsyncEntry` call appends its operation and the
   * constructor arguments (service, account, options) to the file named by `STUB_LOG`. It answers
   * `get` with nothing, `set` with nothing and `delete` with true. The constructor arguments are
   * recorded because the Secret Service pin is an option passed there, so that is the only place
   * it can be observed without a real keyring.
   */
  const recording = `
    import { appendFileSync } from "node:fs";
    export class AsyncEntry {
      constructor(service, account, options) { this.args = { service, account, options }; }
      note(op) { appendFileSync(process.env.STUB_LOG, JSON.stringify({ op, ...this.args }) + "\\n"); }
      async getPassword() { this.note("get"); return null; }
      async setPassword() { this.note("set"); }
      async deletePassword() { this.note("delete"); return true; }
    }`;

  // Left to itself the binding falls back to the kernel keyutils keyring, which loses tokens when the
  // login session ends (see the note on `KeyringBackend`). A store that seems to work and then drops
  // the token is worse than the file fallback, so every entry must carry `store: "secret-service"`.
  it("pins every entry to the Secret Service, never the kernel keyutils keyring", async () => {
    const b = await childWith(recording);
    await b.get("svc", "acct");
    await b.set("svc", "acct", "FAKE");
    await b.delete("svc", "acct");
    const calls = (await readFile(join(tmp.path, "calls.log"), "utf8"))
      .trim()
      .split("\n")
      .map(
        (l) => JSON.parse(l) as { op: string; service: string; account: string; options: unknown },
      );
    expect(calls.map((c) => c.op)).toEqual(["get", "set", "delete"]);
    for (const c of calls)
      expect(c).toMatchObject({
        service: "svc",
        account: "acct",
        options: { linux: { store: "secret-service" } },
      });
  });

  // The constructor is where a missing session bus shows up, before any call is made.
  it("reports an unavailable keyring when the entry cannot be constructed", async () => {
    const b = await childWith(
      `export class AsyncEntry { constructor() { throw new Error("no bus"); } }`,
    );
    await expect(new KeyringStore(b, 10_000).get(baseUrl, "a")).rejects.toThrow(/no bus/);
  });

  it("reports an unavailable keyring when the native module cannot be loaded", async () => {
    const b = await childWith(undefined);
    await expect(new KeyringStore(b, 10_000).get(baseUrl, "a")).rejects.toThrow(
      /could not be loaded/,
    );
  });
});

// The real child (keyring-child.ts run as source) against a fake session bus that accepts the
// connection and never answers: the case that used to hang the CLI for good. The native
// constructor connects to the bus synchronously and has no timeout of its own
// (Brooooooklyn/keyring-node#156), so nothing inside the CLI's process can interrupt it, and a
// worker thread cannot be terminated out of it either. Only killing a child process ends the wait,
// which is why each call runs in one.
describe.skipIf(process.platform !== "linux")("real child against a bus that never answers", () => {
  it("returns 'did not answer' within the timeout plus a margin, and the process exits", async (ctx) => {
    const dir = await mkdtemp(join(process.env.GXY_TEST_ROOT ?? tmpdir(), "gxy-bus-"));
    const sock = join(dir, "bus.sock");
    const sockets = new Set<Socket>();
    // The fake bus: it accepts each connection, reads and discards whatever arrives, and never writes.
    // It keeps the sockets so the `finally` can destroy them, or `server.close` would wait for them.
    const server = createServer((s) => {
      sockets.add(s);
      s.on("data", () => {});
      s.on("error", () => {});
    });
    try {
      try {
        await new Promise<void>((res, rej) => {
          server.once("error", rej);
          server.listen(sock, res);
        });
      } catch {
        // eslint-disable-next-line vitest/no-disabled-tests -- a skip decided at run time, not a disabled test
        ctx.skip(); // no unix sockets here; reported as skipped, not as a pass
      }
      // The store runs in a separate process, so a hang shows as a process that never exits (the 8 s
      // guard below kills it and fails the test) and not as a test that stalls.
      const runner = join(dir, "runner.mjs");
      const store = pathToFileURL(join(__dirname, "../../src/auth/keyring-store.ts")).href;
      await writeFile(
        runner,
        `
        import { KeyringStore } from ${JSON.stringify(store)};
        try { await new KeyringStore(undefined, 800).get("gxy:https://galaxy.example.org", "alice"); console.log("ANSWERED"); }
        catch (e) { console.log(e.message); }`,
      );
      const t0 = Date.now();
      const out = await new Promise<{ code: number | null; stdout: string }>((resolve, reject) => {
        const c = spawn(process.execPath, [runner], {
          env: { PATH: process.env.PATH, HOME: dir, DBUS_SESSION_BUS_ADDRESS: `unix:path=${sock}` },
          stdio: ["ignore", "pipe", "inherit"],
        });
        let stdout = "";
        c.stdout.on("data", (d: Buffer) => (stdout += d.toString()));
        const guard = setTimeout(() => {
          c.kill("SIGKILL");
          reject(new Error("the process did not exit"));
        }, 8000);
        c.on("close", (code) => {
          clearTimeout(guard);
          resolve({ code, stdout });
        });
      });
      expect(out.stdout).toMatch(/did not answer a get within 800 ms/);
      expect(out.code).toBe(0);
      // What this bounds is the wait: the timeout plus a margin for starting two Node processes, not
      // however long the bus stays silent. The margin is wide so a loaded machine does not fail it.
      expect(Date.now() - t0).toBeLessThan(800 + 5000);
    } finally {
      for (const s of sockets) s.destroy();
      server.close();
      await rm(dir, { recursive: true, force: true });
    }
  }, 15000);
});
