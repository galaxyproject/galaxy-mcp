/**
 * Every combination of the places an operation can take its key and its Galaxy from, run through
 * the real resolver (`loadSourcesAsync` and `resolveConnection`, with the exit codes `program.ts`
 * gives their errors) and checked against `expected` below, a plain statement of the rules in the
 * README. The `.env` and `~/.planemo.yml` are real files, so their parsing runs too.
 *
 * Nine dimensions of three values each, 3^9 = 19,683 cases:
 *
 * | # | Source | 0 | 1 | 2 |
 * |---|---|---|---|---|
 * | d1 | `--api-key` | unset | a key | `""` |
 * | d2 | `--url` | unset | a URL | `""` |
 * | d3 | `--profile` | none | `p` | `ghost` (not in the file) |
 * | d4 | `GALAXY_URL` in the environment | unset | a URL | `""` |
 * | d5 | `GALAXY_API_KEY` in the environment | unset | a key | `""` |
 * | d6 | `GALAXY_URL` in `.env` | absent | a URL | `GALAXY_URL=` |
 * | d7 | `GALAXY_API_KEY` in `.env` | absent | a key | `GALAXY_API_KEY=` |
 * | d8 | profile `p` | key and `galaxy_url` | key only | `galaxy_url` only |
 * | d9 | stored login, for the `.env` URL | none | valid | expired |
 *
 * Every source has its own key and URL, so the answer shows which one won. The stored login is
 * for the `.env`'s Galaxy because that is where the subtle rules meet: a `.env` URL picks a stored
 * login's Galaxy, whose token then outranks the `.env` key, while no other key may go there.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AuthError } from "../src/auth/types";
import { CredentialScopeError, loadSources, loadSourcesAsync, resolveConnection, type GlobalFlags } from "../src/config";
import { exitCodeForAuthError } from "../src/exit";
import { FakeStore, fakeLogin, useTempDir } from "./auth/harness";

const K = { flag: "key-flag", env: "key-env", dotenv: "key-dotenv", profile: "key-profile", stored: "key-stored" };
const U = {
  flag: "https://url-flag.example.test",
  env: "https://url-env.example.test",
  dotenv: "https://url-dotenv.example.test",
  profile: "https://url-profile.example.test",
};
const NOW = new Date("2026-06-01T00:00:00.000Z");

/** The value of a dimension: unset, the source's own value, or set to empty. */
const tri = <T>(d: number, value: T): T | "" | undefined => (d === 0 ? undefined : d === 1 ? value : "");

type Outcome =
  | { url: string; key: string; source: "flag" | "profile" | "env" | "stored" | "dotenv" }
  | { exit: 64 | 77; why: string };

/**
 * The rules, as the README states them, written out once more so that a change to the resolver
 * that changes an answer fails here. Empty values count as unset throughout.
 *
 * - An unknown `--profile` is refused (64) before anything else.
 * - The Galaxy: `--url`, the profile's `galaxy_url`, `GALAXY_URL` in the environment, then in
 *   `.env`, then the only stored login. Only the first four count as "named".
 * - The key: `--api-key`, the profile's key, `GALAXY_API_KEY`, the stored login for the Galaxy
 *   picked, then `.env`'s key.
 * - A key other than the stored login's goes only to a named Galaxy (64 otherwise), and never to
 *   one only `.env` named unless it is `.env`'s own key (77).
 * - An expired stored login is refused (77), never replaced by the `.env` key below it.
 */
function expected(d: number[]): Outcome {
  const [apiKey, url, profile, envUrl, envKey, dotUrl, dotKey, shape, stored] = d as [number, number, number, number, number, number, number, number, number];
  if (profile === 2) return { exit: 64, why: "unknown profile" };
  const prof = profile === 1 ? { url: shape !== 1 ? U.profile : undefined, key: shape !== 2 ? K.profile : undefined } : {};

  const hosts: [string, string | undefined][] = [
    ["flag", tri(url, U.flag) || undefined],
    ["profile", prof.url],
    ["env", tri(envUrl, U.env) || undefined],
    ["dotenv", tri(dotUrl, U.dotenv) || undefined],
  ];
  const namedBy = hosts.find(([, v]) => v);
  // The only stored login, when nothing names a Galaxy.
  const picked = namedBy ?? (stored ? (["stored", U.dotenv] as const) : undefined);
  const named = namedBy?.[1];

  const sendTo = (key: string, source: "flag" | "profile" | "env" | "dotenv"): Outcome => {
    if (!named) return { exit: 64, why: source === "profile" ? "profile has no galaxy_url" : "no Galaxy named" };
    if (namedBy?.[0] === "dotenv" && source !== "dotenv") return { exit: 77, why: ".env URL with another key" };
    return { url: named, key, source };
  };
  if (tri(apiKey, K.flag)) return sendTo(K.flag, "flag");
  if (prof.key) return sendTo(prof.key, "profile");
  if (tri(envKey, K.env)) return sendTo(K.env, "env");
  if (stored && picked?.[1] === U.dotenv) {
    if (stored === 2) return { exit: 77, why: "expired login" };
    return { url: U.dotenv, key: K.stored, source: "stored" };
  }
  if (tri(dotKey, K.dotenv)) return sendTo(K.dotenv, "dotenv");
  return { exit: 64, why: "no credentials" };
}

describe("the key and the Galaxy an operation uses, every combination", () => {
  const tmp = useTempDir("gxy-grid-");
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("matches the README's rules in all 3^9 cases", async () => {
    // One working directory per `.env` (d6 x d7) and one HOME per profile shape (d8), made once.
    const cwds = new Map<string, string>();
    for (const dotUrl of [0, 1, 2])
      for (const dotKey of [0, 1, 2]) {
        const dir = join(tmp.path, `cwd-${dotUrl}${dotKey}`);
        mkdirSync(dir);
        const lines = [
          dotUrl === 0 ? "" : `GALAXY_URL=${tri(dotUrl, U.dotenv)}\n`,
          dotKey === 0 ? "" : `GALAXY_API_KEY=${tri(dotKey, K.dotenv)}\n`,
        ];
        writeFileSync(join(dir, ".env"), lines.join(""));
        cwds.set(`${dotUrl}${dotKey}`, dir);
      }
    const homes = [0, 1, 2].map((shape) => {
      const dir = join(tmp.path, `home-${shape}`);
      mkdirSync(dir);
      const p = [
        shape !== 1 ? `    galaxy_url: ${U.profile}\n` : "",
        shape !== 2 ? `    galaxy_user_key: ${K.profile}\n` : "",
      ].join("");
      writeFileSync(join(dir, ".planemo.yml"), `profiles:\n  p:\n${p}`);
      return dir;
    });
    const login = (expiresAt: string | null) =>
      fakeLogin({ baseUrl: U.dotenv as never, token: K.stored, expiresAt });
    const stores = [
      () => new FakeStore(),
      () => new FakeStore().seed(login(null)),
      () => new FakeStore().seed(login("2026-05-01T00:00:00.000Z")),
    ];

    const cwd = vi.spyOn(process, "cwd");
    const mismatches: string[] = [];
    let cases = 0;
    for (let n = 0; n < 3 ** 9; n++) {
      const d = n.toString(3).padStart(9, "0").split("").map(Number);
      const [apiKey, url, profile, envUrl, envKey, dotUrl, dotKey, shape, stored] = d as [number, number, number, number, number, number, number, number, number];
      const flags: GlobalFlags = {
        apiKey: tri(apiKey, K.flag),
        url: tri(url, U.flag),
        profile: profile === 0 ? undefined : profile === 1 ? "p" : "ghost",
      };
      const env: Record<string, string> = {};
      if (envUrl) env.GALAXY_URL = tri(envUrl, U.env)!;
      if (envKey) env.GALAXY_API_KEY = tri(envKey, K.env)!;
      cwd.mockReturnValue(cwds.get(`${dotUrl}${dotKey}`)!);
      vi.stubEnv("HOME", homes[shape]!);
      vi.stubEnv("USERPROFILE", homes[shape]!); // `os.homedir()` reads this one on Windows
      let got: Outcome;
      try {
        // The real readers of `.env` and the profiles, with the environment given here.
        const base = { ...loadSources(), env };
        const src = await loadSourcesAsync(flags, { store: stores[stored]!(), base, now: NOW });
        const conn = resolveConnection(flags, src, "gxy");
        got = { url: conn.baseUrl, key: conn.apiKey, source: conn.keySource as never };
      } catch (e) {
        // As program.ts maps them.
        const exit =
          e instanceof CredentialScopeError ? 77
          : e instanceof AuthError ? exitCodeForAuthError(e.kind)
          : 64;
        got = { exit: exit as 64 | 77, why: (e as Error).message };
      }
      const want = expected(d);
      cases++;
      const same = "exit" in want ? "exit" in got && got.exit === want.exit : !("exit" in got) && got.url === want.url && got.key === want.key && got.source === want.source;
      if (!same && mismatches.length < 20) mismatches.push(`${d.join("")}: want ${JSON.stringify(want)}, got ${JSON.stringify(got)}`);
    }
    vi.unstubAllEnvs();
    expect(cases).toBe(19_683);
    expect(mismatches).toEqual([]);
  }, 120_000);
});
