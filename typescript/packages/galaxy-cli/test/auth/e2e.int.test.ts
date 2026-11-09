/**
 * End to end: the built binary against the fake Galaxy, for what only the real binary shows:
 * the wiring in `deps.ts` and `index.ts`, real processes, real pipes, real files. One journey
 * runs as `gxy`; the rest are cases that need the binary. The config dir and HOME are temp
 * dirs, so nothing here touches the developer's real logins.
 *
 * Every other auth test calls a command function in process, with fakes injected. That cannot
 * show the exit status a shell receives, a real SIGINT, a real EPIPE on a pipe whose reader has
 * gone, or `interactive` being decided from real terminals. The fake Galaxy is a real HTTP server because this binary is a separate
 * process, and cannot see an in-process mock.
 *
 * The CLI is built here into a private cache directory, and galaxy-ops (which the binary
 * imports at run time) is built first, so a stale or missing build cannot be what is tested.
 */
import { afterAll, beforeAll, describe, expect, test as base, vi } from "vitest";
import { execa, type ResultPromise } from "execa";
import { spawnSync } from "node:child_process";
import { chmod, mkdtemp, mkdir, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";
import { createServer, type Socket } from "node:net";
import { fileURLToPath } from "node:url";
import { startFakeGalaxy, type FakeGalaxy } from "../support/fake-galaxy";

const pkgDir = fileURLToPath(new URL("../..", import.meta.url));
// Built into a private directory inside the package (so `galaxy-ops` still resolves from
// node_modules), never into `dist`: `bin.smoke.int.test.ts` reads that.
const outDir = join(pkgDir, "node_modules", ".cache", "gxy-e2e");
// Each name is its own built entry file, as `bin` in package.json has it. The program names itself in
// its hints from the entry it was started through, and `gxy auth login --with-token` must not tell a
// `galaxy-cli` user to run `gxy`. They are started as `node <file>`, with no symlink: that is what
// the pnpm and Windows shims do, so a name that survives it survives every install.
const ENTRIES = { gxy: "gxy.js", "galaxy-cli": "index.js" } as const;
const NAMES = Object.keys(ENTRIES) as (keyof typeof ENTRIES)[];
type Name = (typeof NAMES)[number];

let work: string | undefined;
const bins = Object.fromEntries(NAMES.map((n) => [n, join(outDir, ENTRIES[n])])) as Record<
  Name,
  string
>;

beforeAll(async () => {
  // The binary imports galaxy-ops' build at run time, so build it too rather than trust a cached one.
  // `tsup` is run straight from each package's own `node_modules/.bin` (`preferLocal`): no package
  // manager is involved, so this needs nothing but the installed dependencies (Node 26 no longer
  // ships corepack).
  await execa("tsup", [], { cwd: join(pkgDir, "..", "galaxy-ops"), preferLocal: true });
  await execa("tsup", ["--out-dir", outDir], { cwd: pkgDir, preferLocal: true });
  work = await mkdtemp(join(process.env.GXY_TEST_ROOT ?? tmpdir(), "gxy-e2e-"));
}, 240_000);

afterAll(async () => {
  // `work` is unset when beforeAll failed: leave that failure as the one that is reported.
  if (work) await rm(work, { recursive: true, force: true });
  await rm(outDir, { recursive: true, force: true });
});

type Run = (
  args: string[],
  opts?: {
    env?: Record<string, string>;
    input?: string;
    /** Leave stdin an open pipe that never gets anything, as a terminal does while a person has not typed. */
    openStdin?: boolean;
    name?: Name;
    cwd?: string;
  },
) => ResultPromise<{ reject: false }>;
type Login = Awaited<ReturnType<Run>>;
/** What one run of the CLI printed, for the checks that no secret went to a screen. */
interface Seen {
  args: string[];
  stdout: string;
  stderr: string;
}

// Every test gets its own HOME (and so its own config dir, under it), its own fake Galaxy and a
// `run` that starts the binary with them. They are made when a test asks for them and removed when it ends.
// These are fixtures, not `let`s shared by the file, so the token sweep below sees only this
// test's files and requests, and one test's login cannot answer for another's.
const test = base.extend<{
  home: string;
  configDir: string;
  galaxyWith: () => Promise<FakeGalaxy>;
  galaxy: FakeGalaxy;
  transcript: Seen[];
  cliEnv: (extra?: Record<string, string>) => Record<string, string>;
  run: Run;
  pasteLogin: (g: FakeGalaxy) => Promise<Login>;
}>({
  home: async ({ onTestFinished }, use) => {
    const dir = await mkdtemp(join(work!, "home-"));
    onTestFinished(() => rm(dir, { recursive: true, force: true }));
    await use(dir);
  },
  configDir: async ({ home }, use) => use(join(home, "config")),
  galaxyWith: async ({ onTestFinished }, use) =>
    use(async () => {
      const g = await startFakeGalaxy();
      onTestFinished(() => g.close());
      return g;
    }),
  galaxy: async ({ galaxyWith }, use) => use(await galaxyWith()),
  transcript: async ({}, use) => use([]),
  // The whole environment the binary sees (`extendEnv: false` in `run`): nothing of the
  // developer's `GALAXY_*` or proxy settings reaches it.
  cliEnv: async ({ home, configDir }, use) =>
    use((extra = {}) => ({
      PATH: process.env.PATH ?? "",
      HOME: home,
      GXY_CONFIG_DIR: configDir,
      ...extra,
    })),
  run: async ({ home, cliEnv, transcript }, use) =>
    use((args, opts = {}) => {
      // Typed as text output: `input` and `stdin` below are both conditional, which widens execa's result.
      const child = execa(process.execPath, [bins[opts.name ?? "gxy"], ...args], {
        // The working directory is where `.env` is read from, so by default it is the empty HOME
        // and never the repository.
        cwd: opts.cwd ?? home,
        extendEnv: false,
        env: cliEnv(opts.env),
        input: opts.input,
        // With no input, stdin is /dev/null: a command that reads it sees end of input rather
        // than the developer's terminal.
        stdin: opts.input === undefined && !opts.openStdin ? "ignore" : undefined,
        reject: false,
        timeout: 60_000,
      }) as unknown as ResultPromise<{ reject: false }>;
      // Registered before any caller's `await`, so the transcript is complete when the caller resumes.
      void child.then((r) => transcript.push({ args, stdout: r.stdout, stderr: r.stderr }));
      return child;
    }),
  /** Log in to `g` with a key it has just issued, piped in on stdin; the key is `g.issuedTokens().at(-1)`. */
  pasteLogin: async ({ run }, use) =>
    use((g) =>
      run(["auth", "login", "--host", g.baseUrl, "--with-token"], { input: `${g.issueToken()}\n` }),
    ),
});

// The canary: a token no other value in the run resembles, so finding it anywhere is finding the secret.
const CANARY = "fake-canary-6d3f0a9b2e71";

/**
 * Every file under `root` (as a path relative to it, sorted) whose bytes contain `needle`. It
 * reads every file, not only the ones the CLI is known to write, so a stray log or backup that
 * holds the token is found too.
 */
async function filesHolding(root: string, needle: string): Promise<string[]> {
  const found: string[] = [];
  const walk = async (dir: string): Promise<void> => {
    for (const entry of await readdir(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) await walk(path);
      else if (entry.isFile() && (await readFile(path)).includes(needle))
        found.push(relative(root, path));
    }
  };
  await walk(root);
  return found.sort();
}

/**
 * The token sweep: the token is only where it belongs, wherever it could have gone. The commands'
 * own tests see only what a command wrote to its streams, so a leak to a file or a URL is
 * invisible to them. The canary is the token, so one string search answers each place:
 * - on disk, in `files` (paths under `home`) and nowhere else under HOME, whatever the file is called;
 * - on the wire, in an `x-api-key` or `authorization` header: never in a URL (which proxies and
 *   logs keep), another header or any body, and never on the version probe;
 * - on screen, only for the command whose job is to print it: `auth token`, and only on stdout.
 */
async function expectTokenOnlyWhereAllowed(
  token: string,
  where: { home: string; files: string[]; galaxy: FakeGalaxy; transcript: Seen[] },
): Promise<void> {
  expect(await filesHolding(where.home, token), "files under HOME holding the token").toEqual(
    where.files,
  );
  // A token put in a query string would be percent-encoded, so both spellings are searched for.
  const forms = [token, encodeURIComponent(token)];
  for (const r of where.galaxy.requests) {
    const what = `${r.method} ${r.url}`;
    for (const form of forms) expect(r.url, what).not.toContain(form);
    // The version probe comes before the key is checked, so it carries no key at all.
    const probe = new URL(r.url, "http://x").pathname.endsWith("/api/version");
    for (const [name, value] of Object.entries(r.headers)) {
      // `x-api-key` is where galaxy-ops sends it, and `Authorization` the other header Galaxy
      // was asked to accept it in. Any other header carrying it is a leak.
      if (!probe && (name === "x-api-key" || name === "authorization")) continue;
      expect([value].flat().join("\n"), `${what}, header ${name}`).not.toContain(token);
    }
    expect(r.body, `${what}, body`).not.toContain(token);
  }
  for (const seen of where.transcript) {
    const prints = seen.args[0] === "auth" && seen.args[1] === "token";
    if (!prints) expect(seen.stdout, `stdout of ${seen.args.join(" ")}`).not.toContain(token);
    expect(seen.stderr, `stderr of ${seen.args.join(" ")}`).not.toContain(token);
  }
}

describe("auth, end to end", () => {
  test("logs in, hands out the token, runs an operation, logs out", async ({
    galaxy,
    run,
    configDir,
    home,
    transcript,
  }) => {
    const token = galaxy.issueToken({ accessToken: CANARY });
    const login = await run(["auth", "login", "--host", galaxy.baseUrl, "--with-token"], {
      input: `${token}\n`,
    });
    expect(login.exitCode, login.stderr).toBe(0);
    expect(login.stderr).toContain("Logged in to");
    expect(token).toBe(CANARY);
    expect(login.stdout + login.stderr).not.toContain(token);

    // The directory is private to the user, and so is the file that holds the token.
    expect((await stat(configDir)).mode & 0o777).toBe(0o700);
    expect((await stat(join(configDir, "hosts.yml"))).mode & 0o777).toBe(0o600);

    const tok = await run(["auth", "token"]);
    expect(tok.exitCode).toBe(0);
    expect(tok.stdout.trim()).toBe(token);

    // An existing operation uses the stored login with no flags.
    const user = await run(["--format", "json", "get_user"]);
    expect(user.exitCode, user.stderr).toBe(0);
    expect(JSON.parse(user.stdout)).toMatchObject({
      success: true,
      data: { username: "fake-user" },
    });
    expect(galaxy.requestsTo("/api/users/current").at(-1)!.headers["x-api-key"]).toBe(token);

    // The binary identifies itself on every call of its own.
    for (const r of galaxy.requestsTo("/api/version"))
      expect(r.headers["user-agent"], r.url).toMatch(/^gxy\//);

    // Logged in: the token is in the one file that holds logins, and nowhere else.
    await expectTokenOnlyWhereAllowed(token, {
      home,
      files: ["config/hosts.yml"],
      galaxy,
      transcript,
    });

    const out = await run(["auth", "logout"]);
    expect(out.exitCode, out.stderr).toBe(0);
    // A pasted key is forgotten here, not revoked: Galaxy still accepts it.
    expect(galaxy.isTokenValid(token)).toBe(true);
    expect(await readFile(join(configDir, "hosts.yml"), "utf8")).not.toContain(galaxy.baseUrl);
    // Logged out: it is gone from every file under HOME, not only from the one we know about.
    await expectTokenOnlyWhereAllowed(token, { home, files: [], galaxy, transcript });
    const gone = await run(["auth", "token"]);
    expect(gone.exitCode).toBe(77); // EX_NOPERM: nothing stored, so not logged in
    expect(gone.stdout).toBe("");
  }, 90_000);

  describe("login --with-token", () => {
    test("stores an API key read from stdin", async ({ galaxy, run, home, transcript }) => {
      const token = galaxy.issueToken({ accessToken: CANARY });
      const login = await run(["auth", "login", "--host", galaxy.baseUrl, "--with-token"], {
        input: `${token}\n`,
      });
      expect(login.exitCode, login.stderr).toBe(0);
      expect(login.stderr).toContain("Logged in to");
      expect(login.stdout + login.stderr).not.toContain(token);
      expect((await run(["auth", "token"])).stdout.trim()).toBe(token);
      await expectTokenOnlyWhereAllowed(token, {
        home,
        files: ["config/hosts.yml"],
        galaxy,
        transcript,
      });
    }, 30_000);

    test("refuses a key Galaxy does not know, and stores nothing", async ({
      galaxy,
      run,
      configDir,
    }) => {
      const login = await run(["auth", "login", "--host", galaxy.baseUrl, "--with-token"], {
        input: "fake-not-a-key\n",
      });
      expect(login.exitCode).toBe(77);
      await expect(readFile(join(configDir, "hosts.yml"), "utf8")).rejects.toMatchObject({
        code: "ENOENT",
      });
    }, 30_000);

    test("refuses an admin key, storing nothing, unless --insecure-storage says so", async ({
      galaxy,
      run,
      configDir,
      home,
      transcript,
    }) => {
      const token = galaxy.issueToken({ admin: true, accessToken: CANARY });
      const login = await run(["auth", "login", "--host", galaxy.baseUrl, "--with-token"], {
        input: `${token}\n`,
      });
      expect(login.exitCode, login.stderr).toBe(77);
      expect(login.stderr).toContain("admin rights");
      expect(login.stdout + login.stderr).not.toContain(token);
      await expect(readFile(join(configDir, "hosts.yml"), "utf8")).rejects.toMatchObject({
        code: "ENOENT",
      });
      // Refused, so the token reached no file at all, wherever a store might have put it.
      await expectTokenOnlyWhereAllowed(token, { home, files: [], galaxy, transcript });
    }, 30_000);

    test("keeps an admin key in the file, readable only by the owner, when --insecure-storage says so", async ({
      galaxy,
      run,
      configDir,
      home,
      transcript,
    }) => {
      const token = galaxy.issueToken({ admin: true, accessToken: CANARY });
      const login = await run(
        ["auth", "login", "--host", galaxy.baseUrl, "--with-token", "--insecure-storage"],
        { input: `${token}\n` },
      );
      expect(login.exitCode, login.stderr).toBe(0);
      expect(login.stderr).toContain("admin rights");
      const path = join(configDir, "hosts.yml");
      expect((await stat(path)).mode & 0o777).toBe(0o600);
      await expectTokenOnlyWhereAllowed(token, {
        home,
        files: [relative(home, path)],
        galaxy,
        transcript,
      });
    }, 30_000);
  });

  // The way in is `--with-token`, spelled with whichever name was run: 64 is EX_USAGE.
  test.for(NAMES)(
    "login without --with-token, as %s, exits 64 and says to use it under its own name",
    { timeout: 30_000 },
    async (name, { galaxy, run }) => {
      const login = await run(["auth", "login", "--host", galaxy.baseUrl], { name });
      expect(login.exitCode).toBe(64);
      expect(login.stderr).toContain(`${name} auth login needs --with-token`);
      expect(login.stderr).toContain(`${name} auth login --host <url> --with-token`);
      expect(galaxy.requests).toEqual([]);
    },
  );

  test("asks which Galaxy when two are stored", async ({ galaxy, galaxyWith, run, pasteLogin }) => {
    const second = await galaxyWith();
    expect((await pasteLogin(galaxy)).exitCode).toBe(0);
    expect((await pasteLogin(second)).exitCode).toBe(0);
    const ambiguous = await run(["auth", "token"]);
    // 64 is EX_USAGE, not 77: the command line did not say which Galaxy, so the caller can fix it
    // by saying so (--host, or GALAXY_URL), and nothing is wrong with either login.
    expect(ambiguous.exitCode).toBe(64);
    expect(ambiguous.stdout).toBe("");
    expect(ambiguous.stderr).toContain(galaxy.baseUrl);
    expect(ambiguous.stderr).toContain(second.baseUrl);
    expect((await run(["get_user"])).exitCode).toBe(64);
    expect((await run(["auth", "token", "--host", second.baseUrl])).stdout.trim()).toBe(
      second.issuedTokens()[0],
    );
  }, 90_000);

  // A hosts.yml gxy will not use is the person's file to fix, not a bug in gxy: exit 65 (EX_DATAERR) with
  // the reason, never "Unexpected error" and 70, and the file is left as it was. A login a newer gxy
  // wrote is not "not logged in" (77) or "nothing to remove" (0) for its own Galaxy either.
  describe("a hosts.yml that gxy will not use", () => {
    const newerEntry = (url: string) =>
      `hosts:\n  ${url}:\n    username: u\n    method: pasted\n    storage: vault\n    obtained_at: "2026-11-06T10:44:00.000Z"\n`;

    test("a login written by a newer gxy: token, logout and login for that Galaxy say so and exit 65", async ({
      galaxy,
      run,
      configDir,
    }) => {
      await mkdir(configDir, { mode: 0o700 });
      const path = join(configDir, "hosts.yml");
      await writeFile(path, newerEntry(galaxy.baseUrl), { mode: 0o600 });
      const before = await readFile(path, "utf8");
      const attempts = [
        ["auth", "token", "--host", galaxy.baseUrl],
        ["auth", "logout", "--host", galaxy.baseUrl],
        ["auth", "login", "--host", galaxy.baseUrl, "--with-token"],
      ];
      for (const args of attempts) {
        const r = await run(args, { input: `${galaxy.issueToken()}\n` });
        const what = args.slice(0, 2).join(" ");
        expect(r.exitCode, `${what}: ${r.stderr}`).toBe(65);
        expect(r.stderr, what).toContain("written by a newer version of gxy");
        expect(r.stderr, what).not.toMatch(/Unexpected error|Not logged in|nothing to remove/);
        expect(r.stdout, what).toBe("");
      }
      expect(await readFile(path, "utf8")).toBe(before);
    }, 60_000);

    test("a hosts.yml that others can write: exit 65 naming the file, and nothing is changed", async ({
      galaxy,
      run,
      configDir,
    }) => {
      await mkdir(configDir, { mode: 0o700 });
      const path = join(configDir, "hosts.yml");
      await writeFile(path, `hosts: {}\n`);
      await chmod(path, 0o666);
      const r = await run(["auth", "token", "--host", galaxy.baseUrl]);
      expect(r.exitCode, r.stderr).toBe(65);
      expect(r.stderr).toContain(`Refusing to use ${path}`);
      expect(r.stderr).not.toContain("Unexpected error");
      expect(r.stdout).toBe("");
      expect((await stat(path)).mode & 0o777).toBe(0o666);
    }, 30_000);
  });

  // GALAXY_URL and GALAXY_API_KEY are the variables the Python galaxy-mcp server reads, and an
  // explicit key has always won here (config.ts), so an environment that already works keeps working
  // after a login is stored.
  test("lets GALAXY_API_KEY override the stored login", async ({ galaxy, run, pasteLogin }) => {
    expect((await pasteLogin(galaxy)).exitCode).toBe(0);
    const seen = galaxy.requestsTo("/api/users/current").length;
    const user = await run(["--format", "json", "get_user"], {
      env: { GALAXY_URL: galaxy.baseUrl, GALAXY_API_KEY: "fake-env-key" },
    });
    expect(user.exitCode, user.stderr).toBe(0);
    const keys = galaxy
      .requestsTo("/api/users/current")
      .slice(seen)
      .map((r) => r.headers["x-api-key"]);
    expect(keys).toEqual(["fake-env-key"]);
  }, 60_000);

  test("lets `auth token | head -c1` end quietly with 0, when the output is more than the pipe holds", async ({
    galaxy,
    run,
    cliEnv,
    home,
    configDir,
  }) => {
    // Without epipe.ts the failed write is an uncaught exception: a stack trace on stderr, from a
    // command whose output is a secret. Node ignores SIGPIPE, so a gone reader arrives as an EPIPE
    // error on the write, not as a killed process.
    // A short token fits in the pipe's buffer before `head` leaves, so no write ever fails and
    // nothing is tested. A token of half a megabyte cannot: the write is still going when the
    // reader goes, and it fails with a real EPIPE. (Any printable ASCII is a token: token-shape.ts.)
    const token = "fake-big-token-".padEnd(512 * 1024, "x");
    await mkdir(configDir, { mode: 0o700 });
    await writeFile(
      join(configDir, "hosts.yml"),
      `hosts:\n  ${galaxy.baseUrl}:\n    username: fake-user\n    method: pasted\n    scopes: []\n    expires_at: null\n    obtained_at: "2026-01-01T00:00:00.000Z"\n    storage: file\n    token: ${token}\n`,
      { mode: 0o600 },
    );
    const whole = await run(["auth", "token"]);
    expect(whole.exitCode, whole.stderr).toBe(0);
    expect(whole.stdout.length, "more than a 64 KiB pipe buffer").toBeGreaterThan(256 * 1024);

    // Through bash so the pipe to `head` is a real one, and `PIPESTATUS[0]` is gxy's own exit status,
    // which the pipeline's (`head`'s) would hide. Its stderr goes to a file: a stack trace would be there.
    const errFile = join(home, "epipe-stderr");
    const command = `${JSON.stringify(process.execPath)} ${JSON.stringify(bins.gxy)} auth token 2>${JSON.stringify(errFile)} | head -c1 >/dev/null; echo \${PIPESTATUS[0]}`;
    const r = await execa("bash", ["-c", command], {
      cwd: home,
      extendEnv: false,
      env: cliEnv(),
      timeout: 30_000,
    });
    expect(r.stdout.trim()).toBe("0");
    expect(await readFile(errFile, "utf8")).toBe("");
  }, 60_000);

  // `login` sends a key to the host it picks, so a `.env` in a cloned repository must not be able
  // to choose it (`forLogin` in host.ts). Each case points a
  // `.env` at a real decoy Galaxy and then checks the decoy's request log is empty: "never
  // contacted" is shown by what the decoy received, not by what the command printed.
  describe("login never takes its host from a .env in the working directory", () => {
    /** A working directory whose `.env` names `url` as the Galaxy. */
    async function cwdWithDotenv(home: string, url: string): Promise<string> {
      const dir = await mkdtemp(join(home, "cwd-"));
      await writeFile(join(dir, ".env"), `GALAXY_URL=${url}\n`);
      return dir;
    }

    test("a pasted key with no --host is a usage error, and the .env's Galaxy is never contacted", async ({
      galaxyWith,
      run,
      configDir,
      home,
    }) => {
      const decoy = await galaxyWith();
      const cwd = await cwdWithDotenv(home, decoy.baseUrl);
      const token = decoy.issueToken();
      const login = await run(["auth", "login", "--with-token"], { cwd, input: `${token}\n` });
      expect(login.exitCode, login.stderr).toBe(64);
      expect(decoy.requests).toEqual([]);
      expect(login.stdout + login.stderr).not.toContain(token);
      await expect(readFile(join(configDir, "hosts.yml"), "utf8")).rejects.toMatchObject({
        code: "ENOENT",
      });
    }, 30_000);
  });

  // The command tests abort an `AbortController`; only a real process shows that a real SIGINT
  // reaches it. 130 is 128 plus SIGINT's 2, the shell's convention for a Ctrl-C.
  describe("Ctrl-C", () => {
    // The key is read from stdin, where a real SIGINT is not a character in raw mode but a signal,
    // and the read must end with it: no key was ever read, so nothing is stored and nothing lingers.
    test("aborts the wait for a key on stdin with --with-token: exit 130, nothing stored, and the process is gone", async ({
      galaxy,
      run,
      configDir,
    }) => {
      const child = run(["auth", "login", "--host", galaxy.baseUrl, "--with-token"], {
        openStdin: true,
      });
      try {
        // "Logging in to" is printed before the key is read; once it is out, the read is next.
        let stderr = "";
        child.stderr.on("data", (d: Buffer) => (stderr += d.toString()));
        await vi.waitFor(
          () => {
            expect(stderr).toContain("Logging in to");
          },
          { timeout: 15_000, interval: 20 },
        );
        child.kill("SIGINT");
        const r = await child;
        expect(r.signal).toBeUndefined();
        expect(r.exitCode).toBe(130);
        expect(r.stderr).toContain("Cancelled");
        expect(galaxy.requests).toEqual([]); // no key, so nothing was sent anywhere
        await expect(readFile(join(configDir, "hosts.yml"), "utf8")).rejects.toMatchObject({
          code: "ENOENT",
        });
        expect(child.exitCode).toBe(130); // the process has exited, not been killed by the harness
      } finally {
        child.kill();
        await child;
      }
    }, 30_000);
  });

  // The probe of a Galaxy that accepts the connection and never answers is the one request whose
  // socket outlives the command: after Ctrl-C the login has long said "Cancelled.", but a request
  // left open holds the process until its own timeout (15 s).
  test("Ctrl-C while a Galaxy never answers the probe: exit 130 at once, not after the timeout", async ({
    run,
    configDir,
    onTestFinished,
  }) => {
    const sockets = new Set<Socket>();
    const silent = createServer((c) => {
      sockets.add(c);
      c.on("data", () => {});
      c.on("error", () => {});
    });
    await new Promise<void>((res) => silent.listen(0, "127.0.0.1", res));
    onTestFinished(() => {
      for (const c of sockets) c.destroy();
      silent.close();
    });
    const { port } = silent.address() as { port: number };
    const child = run(["auth", "login", "--host", `http://127.0.0.1:${port}`, "--with-token"], {
      input: "fake-token-0123456789abcdef\n",
    });
    try {
      await vi.waitFor(
        () => {
          if (sockets.size === 0) throw new Error("the CLI never asked the Galaxy anything");
        },
        { timeout: 15_000, interval: 20 },
      );
      const sent = Date.now();
      child.kill("SIGINT");
      const r = await child;
      expect(r.exitCode, r.stderr).toBe(130);
      expect(r.stderr).toContain("Cancelled.");
      expect(Date.now() - sent, "the probe's request was left open").toBeLessThan(5_000);
      await expect(readFile(join(configDir, "hosts.yml"), "utf8")).rejects.toMatchObject({
        code: "ENOENT",
      });
    } finally {
      child.kill();
      await child;
    }
  }, 30_000);

  // `interactive` is decided in deps.ts from the real streams: stdin and stderr both terminals. Only a
  // pseudo-terminal shows it: the in-process login tests fake a terminal with a flag, so nothing else
  // sees this wiring. util-linux `script` gives the binary one without a native addon (`node-pty`
  // would need building). Only util-linux's `script` is used, because the BSD one takes different arguments.
  const scriptProbe = spawnSync("script", ["--version"], { encoding: "utf8" });
  const hasScript =
    process.platform === "linux" &&
    scriptProbe.error === undefined &&
    /util-linux/.test(scriptProbe.stdout);
  describe.skipIf(!hasScript)("on a terminal (a pseudo-terminal from `script`)", () => {
    /**
     * `script` runs `command` with a terminal for stdin, stdout and stderr, and exits with its status
     * (`-e`; `-q` drops its banner and the typescript file is /dev/null). All three streams are the
     * terminal, so what the CLI writes to stderr arrives on `script`'s stdout, and the tests read
     * the screen from that one stream. `stdin: "pipe"` is how a test types Enter.
     */
    function onTerminal(command: string, home: string, env: Record<string, string>) {
      return execa("script", ["-qec", command, "/dev/null"], {
        cwd: home,
        extendEnv: false,
        env,
        stdin: "pipe",
        reject: false,
        timeout: 60_000,
      });
    }
    const quoted = (...args: string[]) => args.map((a) => JSON.stringify(a)).join(" ");

    // With a terminal on stdin the key is typed, and the prompt for it goes to stderr. Read as a
    // pipe the typed key would be echoed on the screen and into scrollback, so with stderr
    // redirected it is refused at once, before a key is read: the leak the silent prompt exists to
    // prevent.
    test("--with-token refuses, and reads nothing, when stdin is a terminal and stderr is a file", async ({
      galaxy,
      home,
      cliEnv,
    }) => {
      const errFile = join(home, "login-stderr");
      const command = `${quoted(process.execPath, bins.gxy, "auth", "login", "--host", galaxy.baseUrl, "--with-token")} 2>${JSON.stringify(errFile)}`;
      const child = onTerminal(command, home, cliEnv());
      try {
        const r = await child;
        expect(r.exitCode, r.stdout).toBe(64);
        expect(await readFile(errFile, "utf8")).toContain("stdin is a terminal but stderr is not");
        expect(galaxy.requests).toEqual([]);
      } finally {
        child.kill();
        await child;
      }
    }, 60_000);
  });
});
