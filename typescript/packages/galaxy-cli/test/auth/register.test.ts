import { describe, it, expect, vi } from "vitest";
import type { Command } from "commander";
import { createGalaxyContext } from "@galaxyproject/galaxy-ops";
import { buildProgram } from "../../src/program";
import { FAKE_TOKEN, HOST, harness } from "./harness";

const HOST_ARG = "https://dev.galaxy.example.org";
const loginArgv = ["auth", "login", "--host", HOST_ARG, "--with-token"];
/** A harness whose stdin holds the key `login --with-token` reads. */
const loggingIn = () => harness({ stdinText: `${FAKE_TOKEN}\n` });

async function run(argv: string[], h = loggingIn()) {
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
  it("are the one subcommand of `auth`: login", () => {
    const auth = buildProgram().commands.find((c) => c.name() === "auth")!;
    expect(auth.commands.map((c) => c.name())).toEqual(["login"]);
  });

  it("run through the injected deps, with the program's name, and set the exit code", async () => {
    const { h, code, makeAuthDeps } = await run(loginArgv);
    expect(makeAuthDeps).toHaveBeenCalledWith("gxy");
    expect(h.store.logins.has(HOST)).toBe(true);
    expect(code).toBe(0);
  });

  // A flag that does not apply is a usage error, not a warning: with `--profile prod --with-token`
  // the key on stdin would otherwise go to a Galaxy other than prod's before anyone could react.
  // The refusal comes before deps are built, stdin is read, anything is sent or stored.
  describe("refuse a global flag that does not apply to login", () => {
    const refusal = async (argv: string[]) => {
      const write = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
      try {
        const result = await run(argv);
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

  // Building the deps opens the credential store. `--help` and the operations
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
    }).parseAsync(["node", "gxy", ...loginArgv]);
    expect(process.exitCode).toBe(70);
    expect(err.mock.calls.flat().join("")).toContain("nope");
    err.mockRestore();
    process.exitCode = 0;
  });
});

describe("options", () => {
  it("--with-token and --insecure-storage reach login", async () => {
    const h = loggingIn();
    const { code } = await run([...loginArgv, "--insecure-storage"], h);
    expect(code).toBe(0);
    expect(h.store.logins.get(HOST)).toMatchObject({ method: "pasted", storage: "file" });
  });
});
