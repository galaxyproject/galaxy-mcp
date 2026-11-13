import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { afterEach, beforeEach, describe, it, expect, vi } from "vitest";
import { createGalaxyContext, invokeWorkflowOp, runWithEnvelope } from "@galaxyproject/galaxy-ops";
import {
  exitCodeFor,
  exitCodeForAuthError,
  EX_USAGE,
  EX_SOFTWARE,
  EX_PROTOCOL,
  EX_UNAVAILABLE,
} from "../src/exit";
import { buildProgram } from "../src/program";
import { useTempDir } from "./auth/harness";

describe("exitCodeFor", () => {
  it("maps error kinds to BSD sysexits", () => {
    expect(exitCodeFor(undefined)).toBe(0);
    expect(exitCodeFor("auth")).toBe(77);
    expect(exitCodeFor("not_found")).toBe(66);
    expect(exitCodeFor("connection")).toBe(69);
    expect(exitCodeFor("version")).toBe(76);
    expect(exitCodeFor("tool_request_rejected")).toBe(65);
    expect(exitCodeFor("job_failed")).toBe(70);
    expect(exitCodeFor("unknown")).toBe(70);
  });
  it("exposes usage + software constants", () => {
    expect(EX_USAGE).toBe(64);
    expect(EX_SOFTWARE).toBe(70);
    expect(EX_PROTOCOL).toBe(76);
  });
  it("keeps a too-old server apart from an unreachable one", () => {
    // Both are the server's fault, but only one is worth retrying.
    expect(exitCodeFor("version")).not.toBe(exitCodeFor("connection"));
  });
});

describe("exitCodeForAuthError", () => {
  // Plain http to a remote host is refused as `insecure`, 76 like a Galaxy's untrustworthy
  // endpoints, and a wrong flag is 64: a script can tell the two apart.
  it("keeps an insecure Galaxy (76) apart from a flag to fix (64) and from no answer (69)", () => {
    expect(exitCodeForAuthError("insecure")).toBe(EX_PROTOCOL);
    expect(exitCodeForAuthError("usage")).toBe(EX_USAGE);
    expect(exitCodeForAuthError("network")).toBe(EX_UNAVAILABLE);
  });

  // The credential file is what cannot be used: 65 (the input is wrong), not 77, which a script
  // reads as "log in" when logging in would meet the same refusal, and not 70, which says gxy
  // has a bug.
  it("keeps a credential file that cannot be used (65) apart from not logged in (77) and a bug (70)", () => {
    expect(exitCodeForAuthError("store")).toBe(65);
    expect(exitCodeForAuthError("store")).not.toBe(exitCodeForAuthError("not_logged_in"));
    expect(exitCodeForAuthError("store")).not.toBe(EX_SOFTWARE);
  });
});

describe("a rejected argument is a usage error, not an outage", () => {
  it("maps validation to EX_USAGE rather than EX_UNAVAILABLE", () => {
    expect(exitCodeFor("validation")).toBe(EX_USAGE);
    expect(exitCodeFor("validation")).not.toBe(EX_UNAVAILABLE);
  });

  it("carries an op's own rejected argument all the way to exit 64", async () => {
    // The whole chain, not the map on its own: invoke_workflow refuses the argument, the
    // envelope carries the kind out, and the CLI turns that kind into a status. Malformed
    // inputs used to arrive here as "connection" and exit 69, which tells an agent to retry
    // something that can never succeed.
    const ctx = createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      fetchImpl: (async () => {
        throw new Error("no request should be made for an argument the op already refused");
      }) as typeof fetch,
    });
    const result = await runWithEnvelope(
      invokeWorkflowOp as never,
      { workflowId: "wf1", inputs: "{not json" } as never,
      ctx,
    );
    expect(result.success).toBe(false);
    expect(result.message).toMatch(/^inputs must be a JSON object string or an object/);
    expect(result.errorKind).toBe("validation");
    expect(exitCodeFor(result.errorKind)).toBe(EX_USAGE);
    expect(exitCodeFor(result.errorKind)).not.toBe(EX_UNAVAILABLE);
  });
});

/** The real credential path of an operation (no makeContext), against a temp config dir and no keyring. */
describe("credential errors exit by kind", () => {
  const tmp = useTempDir("gxy-exit-");
  beforeEach(async () => {
    // safe-env keeps the environment clean; this test also needs its own empty HOME (no
    // ~/.planemo.yml), config dir and working directory (loadSources reads `.env` from it).
    vi.stubEnv("HOME", tmp.path);
    vi.stubEnv("GXY_CONFIG_DIR", join(tmp.path, "cfg"));
    vi.spyOn(process, "cwd").mockReturnValue(tmp.path);
    await mkdir(join(tmp.path, "cfg"), { mode: 0o700 });
  });
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
    process.exitCode = 0;
  });

  const host = (url: string, expires: string | null) =>
    `  ${url}:\n    username: alice\n    method: device\n    scopes: []\n    expires_at: ${expires === null ? "null" : `"${expires}"`}\n    obtained_at: "2026-01-01T00:00:00.000Z"\n    storage: file\n    token: FAKE-TOKEN\n`;

  async function run(hosts: string, file = `hosts:\n${hosts}`, argv: string[] = []) {
    await writeFile(join(tmp.path, "cfg", "hosts.yml"), file, { mode: 0o600 });
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    process.exitCode = 0;
    try {
      await buildProgram().parseAsync(["node", "gxy", ...argv, "get_user"]);
      return { exitCode: process.exitCode, stderr: err.mock.calls.flat().join("\n") };
    } finally {
      err.mockRestore();
    }
  }

  // 77 is EX_NOPERM and 64 EX_USAGE, the same codes `auth` gives for not being logged in and for not
  // saying which Galaxy (program.ts). Every other credential problem stays a usage error, as it was
  // before stored logins existed.
  it("an expired login with no other key exits 77, saying when it expired and how to renew it", async () => {
    const r = await run(host("https://galaxy.example.org", "2020-01-01T00:00:00.000Z"));
    expect(r.exitCode).toBe(77);
    expect(r.stderr).toMatch(/expired at 2020-01-01 00:00 UTC; run (gxy|galaxy-cli) auth login --host https:\/\/galaxy\.example\.org/);
  });
  // The list leaves out a login a newer gxy wrote; for the Galaxy that was asked for, the error says so
  // and still exits 64, instead of only advising a login that would be refused.
  it("names a login a newer gxy wrote for the Galaxy asked for", async () => {
    const r = await run(
      "",
      'hosts:\n  https://galaxy.example.org:\n    username: alice\n    method: pasted\n    storage: vault\n    obtained_at: "2026-01-01T00:00:00.000Z"\n',
      ["--url", "https://galaxy.example.org"],
    );
    expect(r.exitCode).toBe(64);
    expect(r.stderr).toContain("written by a newer version of gxy");
    expect(r.stderr).toContain("https://galaxy.example.org");
  });
  it("several stored logins and no URL exits 64", async () => {
    const r = await run(host("https://a.example.org", null) + host("https://b.example.org", null));
    expect(r.exitCode).toBe(64);
    expect(r.stderr).toContain("Logged in to several Galaxies");
  });
  it("any other credential problem stays 64", async () => {
    const r = await run("  {}\n");
    expect(r.exitCode).toBe(64);
    expect(r.stderr).toMatch(/No Galaxy credentials/);
  });
  it("warns about a broken store and carries on with the other key, without calling it a usage error", async () => {
    // A key in `.env` (the working directory is the temp dir) is used, and the unreadable
    // hosts.yml is a warning: the person must learn the store is broken, not that they typed something wrong.
    await writeFile(join(tmp.path, ".env"), "GALAXY_URL=http://127.0.0.1:1\nGALAXY_API_KEY=FAKE-DOTENV-KEY\n");
    const r = await run("", "hosts: [unclosed");
    expect(r.stderr).toContain("Warning:");
    expect(r.stderr).toContain("hosts.yml");
    expect(r.exitCode).not.toBe(64);
  });
});

// A stored login ranks above a `.env` key, so the person is told when it was the stored one that
// was used, unless they asked for quiet. The host is loopback and the port closed: nothing answers.
describe("which key was used", () => {
  const tmp = useTempDir("gxy-note-");
  beforeEach(async () => {
    vi.stubEnv("HOME", tmp.path);
    vi.stubEnv("GXY_CONFIG_DIR", join(tmp.path, "cfg"));
    vi.spyOn(process, "cwd").mockReturnValue(tmp.path);
    await mkdir(join(tmp.path, "cfg"), { mode: 0o700 });
    await writeFile(
      join(tmp.path, "cfg", "hosts.yml"),
      'hosts:\n  http://127.0.0.1:1:\n    username: alice\n    method: pasted\n    scopes: []\n    expires_at: null\n    obtained_at: "2026-01-01T00:00:00.000Z"\n    storage: file\n    token: FAKE-TOKEN\n',
      { mode: 0o600 },
    );
    await writeFile(join(tmp.path, ".env"), "GALAXY_API_KEY=FAKE-DOTENV-KEY\n");
  });
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
    process.exitCode = 0;
  });

  async function stderrOf(...flags: string[]) {
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      await buildProgram().parseAsync(["node", "gxy", ...flags, "get_user"]);
      return err.mock.calls.flat().join("\n");
    } finally {
      err.mockRestore();
    }
  }

  it("says the stored login was used instead of the .env key", async () => {
    const text = await stderrOf();
    expect(text).toContain("The stored login for http://127.0.0.1:1 ranks above GALAXY_API_KEY in");
    expect(text).toContain(join(tmp.path, ".env"));
    expect(text).not.toContain("FAKE-DOTENV-KEY");
  });
  it("names the directory of a .env that chose the Galaxy", async () => {
    await writeFile(join(tmp.path, ".env"), "GALAXY_URL=http://127.0.0.1:1\n");
    expect(await stderrOf()).toContain(`Using GALAXY_URL from ${join(tmp.path, ".env")}.`);
  });
  it("says nothing with --quiet", async () => {
    expect(await stderrOf("--quiet")).not.toContain("ranks above");
  });
});
