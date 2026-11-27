import { describe, it, expect, vi } from "vitest";
import type { Command } from "commander";
import { createGalaxyContext } from "@galaxyproject/galaxy-ops";
import { buildProgram } from "../../src/program";
import { parseScopes } from "../../src/auth/register";
import { createAuthDeps } from "../../src/auth/deps";
import { FAKE_TOKEN, HOST, fakeLogin, harness } from "./harness";

const HOST_ARG = "https://dev.galaxy.example.org";
const loginArgv = ["auth", "login", "--host", HOST_ARG, "--with-token"];
/** A harness whose stdin holds the key `login --with-token` reads. */
const loggingIn = () => harness({ stdinText: `${FAKE_TOKEN}\n` });

async function run(argv: string[], h = harness({ stored: [fakeLogin()] })) {
  process.exitCode = 0;
  const makeAuthDeps = vi.fn(() => h.deps);
  await buildProgram({ makeAuthDeps, invokedAs: "gxy" }).parseAsync(["node", "gxy", ...argv]);
  const code = process.exitCode;
  process.exitCode = 0;
  return { h, code, makeAuthDeps };
}

/** Make a program throw instead of exit, and keep its help and usage text off the terminal. */
function contained(command: Command): Command {
  command.exitOverride().configureOutput({ writeOut: () => {}, writeErr: () => {} });
  for (const sub of command.commands) contained(sub);
  return command;
}

// The `auth` group as the program wires it: the deps are injected through `makeAuthDeps`, so what is
// tested is the registration (which command, which option reaches it, when deps are built), not the
// commands' behaviour, which their own files cover.
describe("the auth commands", () => {
  it("are the six subcommands of `auth`", () => {
    const auth = buildProgram().commands.find((c) => c.name() === "auth")!;
    expect(auth.commands.map((c) => c.name()).sort()).toEqual([
      "env",
      "headers",
      "login",
      "logout",
      "status",
      "token",
    ]);
  });

  it("run through the injected deps, with the program's name, and set the exit code", async () => {
    const { h, code, makeAuthDeps } = await run(["auth", "token"]);
    expect(makeAuthDeps).toHaveBeenCalledWith("gxy");
    expect(h.cap.out).toBe(`${FAKE_TOKEN}\n`);
    expect(code).toBe(0);
  });

  // A flag that does not apply is a usage error, not a warning: with `--profile prod --with-token`
  // the key on stdin would otherwise go to a Galaxy other than prod's before anyone could react.
  // The refusal comes before deps are built, stdin is read, anything is sent or stored.
  describe("refuse a global flag that does not apply to login", () => {
    const refusal = async (argv: string[]) => {
      const write = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
      try {
        const result = await run(argv, loggingIn());
        return { ...result, text: write.mock.calls.map((c) => String(c[0])).join("") };
      } finally {
        write.mockRestore();
      }
    };
    const PROFILE_MSG =
      "--profile does not apply to `gxy auth login`: login takes its Galaxy from --host, --url or GALAXY_URL in the environment.\n";
    const KEY_MSG =
      "--api-key does not apply to `gxy auth login`: pipe the key in with --with-token instead.\n";

    it.each([
      ["a profile that exists or not", ["--profile", "p"], PROFILE_MSG],
      ["an --api-key", ["--api-key", "fake-flag-key"], KEY_MSG],
      ["an empty --profile", ["--profile", ""], PROFILE_MSG],
      ["an empty --api-key", ["--api-key", ""], KEY_MSG],
    ])("%s: exit 64, nothing read, asked, sent or stored", async (_what, flags, message) => {
      for (const quiet of [[], ["--quiet"]]) {
        const { h, code, makeAuthDeps, text } = await refusal([...quiet, ...flags, ...loginArgv]);
        expect(code).toBe(64);
        expect(text).toBe(message);
        expect(makeAuthDeps).not.toHaveBeenCalled();
        expect(h.whoamiCalls).toEqual([]);
        expect(h.store.logins.size).toBe(0);
        expect(h.stdin.readableLength).toBeGreaterThan(0); // the key is still waiting on stdin
        expect(h.cap.out + h.cap.err).toBe("");
      }
    });

    it("say both when both flags are given", async () => {
      const { code, text } = await refusal(["--profile", "p", "--api-key", "K", ...loginArgv]);
      expect(code).toBe(64);
      expect(text).toBe(PROFILE_MSG + KEY_MSG);
    });

    it("do not stop login without them, and --quiet alone says nothing", async () => {
      const { h, code, text } = await refusal(["--quiet", ...loginArgv]);
      expect(code).toBe(0);
      expect(text).toBe("");
      expect(h.store.logins.has(HOST)).toBe(true);
    });
  });

  // The other auth commands: `--profile` finds their host (an unknown name is refused by
  // `requireProfile`, not here), and `--api-key` is ignored by all of them but `status`, so it is
  // refused too. `status` reports on the key operations would use, `--api-key` included.
  describe("refuse --api-key for the commands that never use it", () => {
    it.each([["token"], ["logout"], ["env"], ["headers"]])(
      "auth %s: exit 64, nothing run, even with --quiet",
      async (cmd) => {
        const write = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
        try {
          for (const quiet of [[], ["--quiet"]]) {
            write.mockClear();
            const { h, code, makeAuthDeps } = await run([...quiet, "--api-key", "K", "auth", cmd]);
            expect(code).toBe(64);
            expect(write.mock.calls.map((c) => String(c[0])).join("")).toBe(
              `--api-key does not apply to \`gxy auth ${cmd}\`.\n`,
            );
            expect(makeAuthDeps).not.toHaveBeenCalled();
            expect(h.cap.out).toBe("");
            expect(h.store.logins.size).toBe(1); // the stored login is still there
          }
        } finally {
          write.mockRestore();
        }
      },
    );

    // `headers` takes its Galaxy only from CLAUDE_CODE_MCP_SERVER_URL, so a --profile there could
    // only mislead: refused like login's, for any name, before anything is read.
    it("refuse --profile for auth headers: exit 64, nothing run, even with --quiet", async () => {
      const write = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
      try {
        for (const quiet of [[], ["--quiet"]]) {
          write.mockClear();
          const { h, code, makeAuthDeps } = await run([...quiet, "--profile", "p", "auth", "headers"]);
          expect(code).toBe(64);
          expect(write.mock.calls.map((c) => String(c[0])).join("")).toBe(
            "--profile does not apply to `gxy auth headers`: it takes its Galaxy from CLAUDE_CODE_MCP_SERVER_URL.\n",
          );
          expect(makeAuthDeps).not.toHaveBeenCalled();
          expect(h.cap.out).toBe("");
        }
      } finally {
        write.mockRestore();
      }
    });

    it("leave --profile to the commands that use it to find the host", async () => {
      for (const cmd of ["token", "logout", "env", "status"]) {
        const { code, makeAuthDeps } = await run(["--profile", "p", "auth", cmd]);
        expect(code).toBe(0);
        expect(makeAuthDeps).toHaveBeenCalled();
      }
    });

    it("leave --api-key to status, which reports on it, even with --quiet", async () => {
      const write = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
      try {
        for (const quiet of [[], ["--quiet"]]) {
          const { code, makeAuthDeps } = await run([...quiet, "--api-key", "K", "auth", "status"]);
          expect(code).toBe(0);
          expect(makeAuthDeps).toHaveBeenCalled();
        }
        expect(write.mock.calls.map((c) => String(c[0])).join("")).toBe("");
      } finally {
        write.mockRestore();
      }
    });
  });

  // The login warning that a GALAXY_API_KEY outranks the stored login is the command's own (a
  // warning, not a refusal), so `--quiet` has to reach the command.
  it("drop login's GALAXY_API_KEY precedence warning with --quiet", async () => {
    const login = async (...global: string[]) => {
      const h = harness({ env: { GALAXY_API_KEY: "irrelevant" }, stdinText: `${FAKE_TOKEN}\n` });
      await run([...global, "auth", "login", "--with-token", "--host", HOST_ARG], h);
      return h.cap.err;
    };
    expect(await login()).toContain("GALAXY_API_KEY is set and will take precedence");
    expect(await login("--quiet")).not.toContain("GALAXY_API_KEY is set");
  });

  // `login` refuses any --profile, one that is not in ~/.planemo.yml too, before it looks the name
  // up: the refusal is about login, not about the profile. `token` and `logout` do use the profile to
  // find the host, so for them an unknown name is `requireProfile`'s refusal. The host rule here is
  // the real one (`createAuthDeps`), over the harness's fake store, whoami and stdin.
  it.each([
    [
      "login",
      ["auth", "login", "--with-token", "--host", HOST_ARG],
      "does not apply to `gxy auth login`",
    ],
    ["token", ["auth", "token"], "Unknown --profile"],
    ["logout", ["auth", "logout"], "Unknown --profile"],
    ["env", ["auth", "env"], "Unknown --profile"],
    // `headers` takes its Galaxy only from the server Claude Code names, so --profile is refused
    // outright, as for login.
    ["headers", ["auth", "headers"], "does not apply to `gxy auth headers`"],
  ])("refuse an unknown --profile for auth %s with exit 64", async (cmd, argv, message) => {
    const h = harness({ stored: [fakeLogin()], stdinText: `${FAKE_TOKEN}\n` });
    const write = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    process.exitCode = 0;
    let printed = "";
    try {
      const program = buildProgram({
        invokedAs: "gxy",
        makeAuthDeps: (name) => ({
          ...h.deps,
          resolveHost: createAuthDeps(name, () => program.opts()).resolveHost,
        }),
      });
      await program.parseAsync(["node", "gxy", "--profile", "typo", ...argv]);
      printed = write.mock.calls.map((c) => String(c[0])).join("");
    } finally {
      write.mockRestore();
    }
    const code = process.exitCode;
    process.exitCode = 0;
    expect(code).toBe(64);
    expect(printed + h.cap.err).toContain(message);
    expect(h.cap.out).toBe("");
    expect(h.whoamiCalls).toEqual([]);
    expect(await h.store.list()).toHaveLength(1); // still only the seeded login
    if (cmd === "login") expect(h.stdin.readableLength).toBeGreaterThan(0);
  });

  // `status` lists every stored login and resolves no host unless `--host` is given, so it needs the
  // profile check of its own (`checkProfile`, the real one from `createAuthDeps`): before it lists or
  // checks anything, with or without `--host`, and with `--json` nothing goes to stdout.
  it.each([
    ["no --host", []],
    ["--host", ["--host", HOST_ARG]],
    ["--json", ["--json"]],
    ["--json and --host", ["--json", "--host", HOST_ARG]],
  ])("refuses an unknown --profile for auth status with exit 64 (%s)", async (_name, extra) => {
    const h = harness({ stored: [fakeLogin()] });
    const write = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    process.exitCode = 0;
    let printed = "";
    try {
      const program = buildProgram({
        invokedAs: "gxy",
        makeAuthDeps: (name) => {
          const real = createAuthDeps(name, () => program.opts());
          return {
            ...h.deps,
            resolveHost: (explicit, purpose) => real.resolveHost(explicit, purpose),
            checkProfile: () => real.checkProfile?.(),
          };
        },
      });
      await program.parseAsync(["node", "gxy", "--profile", "typo", "auth", "status", ...extra]);
      printed = write.mock.calls.map((c) => String(c[0])).join("");
    } finally {
      write.mockRestore();
    }
    const code = process.exitCode;
    process.exitCode = 0;
    expect(code).toBe(64);
    expect(printed + h.cap.err).toContain("Unknown --profile typo");
    expect(h.cap.out).toBe("");
    expect(h.whoamiCalls).toEqual([]);
  });

  // Building the deps opens the credential store, and with it the keyring. `--help` and the operations
  // must never do that (register.ts calls `makeDeps` once per invocation of an `auth` command).
  it("do not build their deps for --help or for an operation", async () => {
    const makeAuthDeps = vi.fn(() => harness().deps);
    const fetchImpl = (async () => Response.json([])) as typeof fetch;
    const makeContext = () =>
      createGalaxyContext({ baseUrl: "https://g.example", apiKey: "K", fetchImpl });
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    try {
      for (const argv of [["--help"], ["auth", "--help"], ["get_histories"]]) {
        const program = contained(buildProgram({ makeAuthDeps, makeContext, invokedAs: "gxy" }));
        await program.parseAsync(["node", "gxy", ...argv]).catch(() => undefined); // --help ends in a thrown exit
      }
    } finally {
      log.mockRestore();
      process.exitCode = 0;
    }
    expect(makeAuthDeps).not.toHaveBeenCalled();
  });

  // 70 is EX_SOFTWARE: failing to build the wiring is a fault in the CLI, not something the person typed.
  it("report a deps factory that throws as exit 70, with its message", async () => {
    const err = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    process.exitCode = 0;
    await buildProgram({
      makeAuthDeps: () => {
        throw new Error("nope");
      },
    }).parseAsync(["node", "gxy", "auth", "token"]);
    expect(process.exitCode).toBe(70);
    expect(err.mock.calls.flat().join("")).toContain("nope");
    err.mockRestore();
    process.exitCode = 0;
  });
});

describe("options", () => {
  it("--host, --json and --show-token reach status", async () => {
    const { h } = await run(["auth", "status", "--host", HOST_ARG, "--json", "--show-token"]);
    expect((JSON.parse(h.cap.out) as { token?: string }[])[0]?.token).toBe(FAKE_TOKEN);
  });

  it("--shell reaches env, and a value that is not a shell is refused", async () => {
    const { h } = await run(["auth", "env", "--shell", "fish"]);
    expect(h.cap.out).toContain("set -gx GALAXY_API_KEY");

    // One validator: `envCommand`'s, so the exit is the usage code 64 and not commander's 1.
    const bad = await run(["auth", "env", "--shell", "csh"]);
    expect(bad.code).toBe(64);
    expect(bad.h.cap.out).toBe("");
    expect(bad.h.cap.err).toContain('Unknown shell "csh"');
  });

  it("--scopes accumulates across uses, commas and spaces, and reaches the device flow", async () => {
    const h = harness();
    await run(
      ["auth", "login", "--host", HOST_ARG, "--scopes", "a", "--scopes", "b,c", "--scopes", "a d"],
      h,
    );
    expect(h.server.startedWith).toEqual([["a", "b", "c", "d"]]);
  });

  it("--with-token and --insecure-storage reach login", async () => {
    const h = loggingIn();
    const { code } = await run([...loginArgv, "--insecure-storage"], h);
    expect(code).toBe(0);
    expect(h.store.logins.get(HOST)).toMatchObject({ method: "pasted", storage: "file" });
  });

  it("--no-browser stops login offering to open one", async () => {
    const offered = async (extra: string[]) => {
      const h = harness({ interactive: true });
      await run(["auth", "login", "--host", HOST_ARG, ...extra], h);
      return h.cap.err.includes("Press Enter");
    };
    expect(await offered([])).toBe(true);
    expect(await offered(["--no-browser"])).toBe(false);
  });
});

describe("parseScopes", () => {
  it("splits on commas and whitespace", () => {
    expect(parseScopes("a,b")).toEqual(["a", "b"]);
    expect(parseScopes("a b")).toEqual(["a", "b"]);
    expect(parseScopes("a, b  c")).toEqual(["a", "b", "c"]);
  });

  it("adds to what was given before, in order, without repeats", () => {
    expect(parseScopes("b", parseScopes("a"))).toEqual(["a", "b"]);
    expect(parseScopes("b c,a", ["a", "b"])).toEqual(["a", "b", "c"]);
  });
});

/** Listeners on each signal before any command ran. */
const before = {
  SIGINT: process.listenerCount("SIGINT"),
  SIGTERM: process.listenerCount("SIGTERM"),
  SIGHUP: process.listenerCount("SIGHUP"),
};

// Login and logout end on SIGTERM and SIGHUP as they do on Ctrl-C, with the code a shell reports
// for the signal. The default for those two is to die at once, which would skip the revocation of a
// token that was just issued. `process.emit` stands in for the signal: what is tested is that the
// handlers are registered, abort the command and set the code, then are removed.
describe("interrupting login and logout", () => {
  const never = () => new Promise<never>(() => {});

  it.each([
    ["SIGINT", 130],
    ["SIGTERM", 143],
    ["SIGHUP", 129],
  ] as const)(
    "login takes the token back and exits %s's code (%i) when it lands during whoami",
    async (sig, expected) => {
      const h = harness();
      h.deps.whoami = () => (process.emit(sig), never());
      const { code } = await run(["auth", "login", "--host", HOST_ARG], h);
      expect(code).toBe(expected);
      expect(h.server.revoked).toEqual([FAKE_TOKEN]);
      expect(h.cap.err).toContain("Cancelled; the token");
      expect(process.listenerCount(sig)).toBe(before[sig]);
    },
  );

  it.each([
    ["SIGTERM", 143],
    ["SIGHUP", 129],
  ] as const)(
    "logout stops waiting for the revocation on %s, deletes locally, exits %i",
    async (sig, expected) => {
      const h = harness({ stored: [fakeLogin()] });
      h.server.revoke = () => (process.emit(sig), never());
      const { code } = await run(["auth", "logout", "--host", HOST_ARG], h);
      expect(code).toBe(expected);
      expect(await h.store.get(HOST)).toBeNull();
      expect(h.cap.err).toContain("Could not revoke the token on the server");
    },
  );
});
