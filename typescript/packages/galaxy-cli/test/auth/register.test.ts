import { describe, it, expect, vi } from "vitest";
import type { Command } from "commander";
import { createGalaxyContext } from "@galaxyproject/galaxy-ops";
import { buildProgram } from "../../src/program";
import { parseScopes } from "../../src/auth/register";
import { FAKE_TOKEN, HOST, fakeLogin, harness } from "./harness";

const HOST_ARG = "https://dev.galaxy.example.org";

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

  // The global `--api-key` is for the operations. `auth` never reads it, and saying so is better than
  // letting someone think a key they typed was used.
  it("warn once that the global --api-key is ignored, and not with --quiet", async () => {
    const warned = async (...argv: string[]) => {
      const write = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
      try {
        await run(["--api-key", "fake-flag-key", ...argv, "auth", "token"]);
        return write.mock.calls.map((c) => String(c[0])).join("");
      } finally {
        write.mockRestore();
      }
    };
    const text = await warned();
    expect(text).toBe("Warning: --api-key is ignored by `gxy auth` commands.\n");
    expect(await warned("--quiet")).toBe("");
  });

  // `status` reports on the key operations would use, `--api-key` included, so it is not told the flag is ignored.
  it("do not warn that --api-key is ignored for status, which reports on it", async () => {
    const write = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    try {
      await run(["--api-key", "fake-flag-key", "auth", "status"]);
      expect(write.mock.calls.map((c) => String(c[0])).join("")).toBe("");
    } finally {
      write.mockRestore();
    }
  });

  // A login never takes its host from a profile, so `--profile` does nothing there; the other
  // subcommands do use it (host, and which key `status` reports), so they stay quiet.
  it("warn once that the global --profile is ignored by login only, and not with --quiet", async () => {
    const warned = async (...argv: string[]) => {
      const write = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
      try {
        await run(
          ["--profile", "p", ...argv],
          harness({ stored: [fakeLogin()], stdinText: `${FAKE_TOKEN}\n` }),
        );
        return write.mock.calls.map((c) => String(c[0])).join("");
      } finally {
        write.mockRestore();
      }
    };
    const warning =
      "Warning: --profile is ignored by `gxy auth login`: it only uses --host or GALAXY_URL from the environment.\n";
    expect(await warned("auth", "login", "--host", "https://g.example", "--with-token")).toContain(
      warning,
    );
    expect(
      await warned("--quiet", "auth", "login", "--host", "https://g.example", "--with-token"),
    ).not.toContain("--profile");
    expect(await warned("auth", "token")).toBe("");
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

    const program = contained(buildProgram({ makeAuthDeps: () => harness().deps }));
    await expect(
      program.parseAsync(["node", "gxy", "auth", "env", "--shell", "csh"]),
    ).rejects.toThrow();
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
    const h = harness({ stdinText: `${FAKE_TOKEN}\n` });
    const { code } = await run(
      ["auth", "login", "--host", HOST_ARG, "--with-token", "--insecure-storage"],
      h,
    );
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
