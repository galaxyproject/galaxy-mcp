import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, it, expect, vi } from "vitest";
import { loadSources, loadSourcesAsync, resolveConnection, type GlobalFlags, type Sources } from "../src/config";
import { normalizeBaseUrl } from "../src/auth/base-url";
import { AuthError, type StoredLogin } from "../src/auth/types";
import { DAY, FAKE_TOKEN, FakeStore, fakeLogin, useTempDir } from "./auth/harness";

describe("resolveConnection", () => {
  it("prefers explicit flags over env", () => {
    const conn = resolveConnection(
      { url: "https://flag.example", apiKey: "FLAGKEY" },
      { env: { GALAXY_URL: "https://env.example", GALAXY_API_KEY: "ENVKEY" }, dotenv: {}, profiles: {} },
    );
    expect(conn).toMatchObject({ baseUrl: "https://flag.example", apiKey: "FLAGKEY" });
  });
  // `--profile` is a pair, key and URL: its URL ranks right after --url, ahead of GALAXY_URL and .env's,
  // so that its key goes to its own Galaxy (config.ts). The key is the profile's, too.
  it("takes the --profile URL and key over the environment's and .env's URL", () => {
    const profiles = { prod: { galaxy_url: "https://prof.example", galaxy_user_key: "PK" } };
    const dotenv = { GALAXY_URL: "https://dot.example" };
    expect(resolveConnection({ profile: "prod" }, { env: {}, dotenv, profiles })).toMatchObject({ baseUrl: "https://prof.example", apiKey: "PK" });
    // Without --profile the profile is not read at all: .env's URL is used.
    expect(resolveConnection({}, { env: {}, dotenv: { ...dotenv, GALAXY_API_KEY: "DK" }, profiles })).toMatchObject({ baseUrl: "https://dot.example", apiKey: "DK" });
  });
  it("sends the profile's key to the profile's URL, not to a hostile .env or GALAXY_URL", () => {
    const profiles = { p: { galaxy_url: "https://prof.example", galaxy_user_key: "PK" } };
    const hostile = { GALAXY_URL: "https://evil.example" };
    expect(resolveConnection({ profile: "p" }, { env: {}, dotenv: hostile, profiles })).toMatchObject({ baseUrl: "https://prof.example", apiKey: "PK" });
    // The environment names a Galaxy too, and the profile's URL still wins.
    expect(resolveConnection({ profile: "p" }, { env: { GALAXY_URL: "https://env.example" }, dotenv: hostile, profiles })).toMatchObject({ baseUrl: "https://prof.example", apiKey: "PK" });
  });
  // `--url X --profile p` asks for both on the command line, so p's key goes to X.
  it("lets --url beat the profile URL, and p's key goes to it", () => {
    const profiles = { p: { galaxy_url: "https://prof.example", galaxy_user_key: "PK" } };
    expect(resolveConnection({ profile: "p", url: "https://x.example" }, { env: {}, dotenv: {}, profiles })).toMatchObject({ baseUrl: "https://x.example", apiKey: "PK" });
  });
  // A profile with a key and no URL takes the Galaxy from --url or the environment, but never from a .env
  // (a cloned repository must not choose where a planemo key goes), and never from the stored login.
  it("sends a URL-less profile's key to an environment or --url Galaxy, but refuses a .env one", () => {
    const profiles = { p: { galaxy_user_key: "PK" } };
    expect(resolveConnection({ profile: "p" }, { env: { GALAXY_URL: "https://env.example" }, dotenv: {}, profiles })).toMatchObject({ baseUrl: "https://env.example", apiKey: "PK" });
    expect(() => resolveConnection({ profile: "p" }, { env: {}, dotenv: { GALAXY_URL: "https://dot.example" }, profiles, dotenvDir: "/work/repo" }))
      .toThrow(/not sending the profile's key there.*\/work\/repo\/\.env|\/work\/repo\/\.env.*not sending/s);
  });
  it("throws a clear error when no credentials resolve", () => {
    expect(() => resolveConnection({}, { env: {}, dotenv: {}, profiles: {} })).toThrow(/GALAXY_URL|GALAXY_API_KEY/);
  });
});

const tmp = useTempDir("gxy-cfg-");

// A stored login (`gxy auth login`) is one more source for an operation. An explicit key still wins,
// as it did before stored logins existed, and the stored token goes only to the Galaxy it was stored for.
describe("resolveConnection with stored logins", () => {
  const url = normalizeBaseUrl("https://galaxy.example.org");
  const login = (over: Partial<StoredLogin> = {}): StoredLogin => fakeLogin({ baseUrl: url, token: "STORED", expiresAt: "2030-01-01T00:00:00.000Z", ...over });
  const now = new Date("2026-06-01T00:00:00.000Z");
  const src = (over: Partial<Sources> = {}): Sources => ({
    env: {}, dotenv: {}, profiles: {}, storedBaseUrls: [url], storedLogin: login(), now, ...over,
  });
  const at = (ms: number) => new Date(now.getTime() + ms).toISOString();

  it("uses the only stored login when nothing names a host", () => {
    expect(resolveConnection({}, src())).toMatchObject({ baseUrl: url, apiKey: "STORED" });
  });
  it("lets --api-key and GALAXY_API_KEY win", () => {
    const named = { dotenv: { GALAXY_URL: url } };
    expect(resolveConnection({ apiKey: "FLAG" }, src(named)).apiKey).toBe("FLAG");
    expect(resolveConnection({}, src({ ...named, env: { GALAXY_API_KEY: "ENV" } })).apiKey).toBe("ENV");
  });
  it("prefers the stored login over .env and profile keys", () => {
    expect(resolveConnection({}, src({ dotenv: { GALAXY_API_KEY: "DOT" } })).apiKey).toBe("STORED");
  });
  // A `--url` naming another Galaxy finds no stored login, so nothing stored is sent there.
  it("ignores a stored login for a different Galaxy", () => {
    const conn = resolveConnection({ url: "https://other.example" }, src({ dotenv: { GALAXY_API_KEY: "DOT" } }));
    expect(conn).toMatchObject({ baseUrl: "https://other.example", apiKey: "DOT" });
  });
  // The key goes to the vetted form of the URL it was stored for, not to the raw string, which may
  // carry a query or an odd path and merely normalise to the same Galaxy.
  it("matches the URL after normalisation, and sends the key to the normalised URL", () => {
    const conn = resolveConnection({ url: "https://GALAXY.example.org/?x=1" }, src());
    expect(conn).toMatchObject({ baseUrl: url, apiKey: "STORED" });
  });
  it("keeps the raw URL when an explicit key is used", () => {
    expect(resolveConnection({ url: "https://x.example.org/?q", apiKey: "K" }, { env: {}, dotenv: {}, profiles: {} }).baseUrl).toBe("https://x.example.org/?q");
  });
  it("fails with ambiguous_host when several are stored and none named", () => {
    expect(() => resolveConnection({}, src({ storedBaseUrls: [url, normalizeBaseUrl("https://b.example")], storedLogin: undefined })))
      .toThrow(expect.objectContaining({ kind: "ambiguous_host", message: expect.stringMatching(/several Galaxies/) }));
  });
  it("reports a store problem when it leaves no credentials", () => {
    expect(() => resolveConnection({}, { env: {}, dotenv: {}, profiles: {}, storeProblem: "hosts.yml is not valid" }))
      .toThrow(/hosts\.yml is not valid/);
  });

  describe("an expired login", () => {
    it("fails when there is no other key, naming the time and how to renew", () => {
      const s = src({ storedLogin: login({ expiresAt: "2026-05-01T00:00:00.000Z" }) });
      expect(() => resolveConnection({}, s)).toThrow(expect.objectContaining({ kind: "not_logged_in", message: expect.stringMatching(/expired at 2026-05-01.*gxy auth login --host https:\/\/galaxy\.example\.org/) }));
    });
    // An unreadable expiry counts as expired, and so does one within a minute of running out (config.ts).
    it("is used up to the margin: 30 s left counts as about to expire, two minutes does not", () => {
      expect(() => resolveConnection({}, src({ storedLogin: login({ expiresAt: at(30_000) }) }))).toThrow(/about to expire/);
      expect(resolveConnection({}, src({ storedLogin: login({ expiresAt: at(120_000) }) })).apiKey).toBe("STORED");
    });
    it("hands over to another key with a warning", () => {
      const s = src({ storedLogin: login({ expiresAt: "2026-05-01T00:00:00.000Z" }), dotenv: { GALAXY_URL: url, GALAXY_API_KEY: "DOT" } });
      const conn = resolveConnection({}, s);
      expect(conn.apiKey).toBe("DOT");
      expect(conn.warning).toMatch(/expired/);
    });
  });

  describe("an unreadable expiry", () => {
    it("is not read as non-expiring", () => {
      expect(() => resolveConnection({}, src({ storedLogin: login({ expiresAt: "tomorrow" }) }))).toThrow(/unreadable expiry/);
    });
    it("hands over to another key with a warning", () => {
      const conn = resolveConnection({}, src({ storedLogin: login({ expiresAt: "tomorrow" }), dotenv: { GALAXY_URL: url, GALAXY_API_KEY: "OTHER" } }));
      expect(conn.apiKey).toBe("OTHER");
      expect(conn.warning).toMatch(/unreadable expiry/);
    });
  });
});

// Which key may go where. A stored token goes only to the host it was stored for; any other key goes
// only to a host somebody named, never to the host that the stored logins merely supplied (the only
// stored login, when nothing names one). Every combination is run and held to those two rules, then
// the cases that carry the precedence are spelled out.
describe("resolveConnection: keys and hosts", () => {
  const url = normalizeBaseUrl("https://galaxy.example.org");
  const now = new Date("2026-06-01T00:00:00.000Z");
  const stored = {
    valid: fakeLogin({ baseUrl: url, token: "STORED", expiresAt: "2030-01-01T00:00:00.000Z" }),
    expired: fakeLogin({ baseUrl: url, token: "STORED", expiresAt: "2026-05-01T00:00:00.000Z" }),
    missing: undefined,
  };
  // A host source names some other Galaxy than the stored login's, or none at all (then the stored
  // login's is the only candidate): the invariants below are what keep the other keys off it.
  const hosts = {
    none: { flags: {}, src: {} },
    flag: { flags: { url: "https://flag.example" }, src: {} },
    env: { flags: {}, src: { env: { GALAXY_URL: "https://env.example" } } },
    dotenv: { flags: {}, src: { dotenv: { GALAXY_URL: "https://dot.example" } } },
    profile: { flags: { profile: "p" }, src: { profiles: { p: { galaxy_url: "https://prof.example" } } } },
  };
  const keys = {
    none: {},
    env: { env: { GALAXY_API_KEY: "ENVKEY" } },
    dotenv: { dotenv: { GALAXY_API_KEY: "DOTKEY" } },
    profile: { profiles: { p: { galaxy_user_key: "PROFKEY" } } },
  };
  const merge = (...parts: Partial<Sources>[]): Sources => {
    const out: Sources = { env: {}, dotenv: {}, profiles: {}, now };
    for (const p of parts) {
      out.env = { ...out.env, ...p.env };
      out.dotenv = { ...out.dotenv, ...p.dotenv };
      out.profiles = { p: { ...out.profiles.p, ...p.profiles?.p } };
      if (p.dotenvDir) out.dotenvDir = p.dotenvDir;
    }
    return out;
  };

  const rows: [string, string, string][] = [];
  for (const h of Object.keys(hosts)) for (const k of Object.keys(keys)) for (const l of Object.keys(stored)) rows.push([h, k, l]);

  it.each(rows)("host from %s, key from %s, stored login %s: no key goes to a host it was not meant for", (h, k, l) => {
    const host = hosts[h as keyof typeof hosts];
    const login = stored[l as keyof typeof stored];
    const profileFlag = k === "profile" ? { profile: "p" } : {};
    const flags = { ...host.flags, ...profileFlag };
    const s: Sources = {
      ...merge(host.src, keys[k as keyof typeof keys]),
      storedBaseUrls: [url],
      ...(login ? { storedLogin: login } : {}),
    };
    let conn;
    try {
      conn = resolveConnection(flags, s);
    } catch {
      return; // refusing is always allowed
    }
    const hostNamed = h !== "none";
    if (conn.apiKey === "STORED") expect(conn.baseUrl).toBe(url);
    else expect(hostNamed).toBe(true);
    if (!hostNamed) {
      expect(conn).toMatchObject({ apiKey: "STORED", baseUrl: url });
      expect(k === "none" || k === "dotenv").toBe(true); // an env or profile key does not get this far
    }
  });

  const run = (flags: GlobalFlags, ...parts: Partial<Sources>[]) =>
    resolveConnection(flags, { ...merge(...parts), storedBaseUrls: [url], storedLogin: stored.valid });
  const asExpired = (flags: GlobalFlags, ...parts: Partial<Sources>[]) =>
    resolveConnection(flags, { ...merge(...parts), storedBaseUrls: [url], storedLogin: stored.expired });

  describe("a key that is not the stored one needs a named host", () => {
    it("a .env key with no URL is not sent to the only stored login, valid or expired", () => {
      for (const login of [stored.valid, stored.expired, undefined]) {
        const s = { ...merge(keys.dotenv), storedBaseUrls: [url], ...(login ? { storedLogin: login } : {}) };
        // With a valid login the stored token is what is sent (to its own host); the .env key never is.
        if (login === stored.valid) expect(resolveConnection({}, s)).toMatchObject({ apiKey: "STORED", baseUrl: url });
        else expect(() => resolveConnection({}, s)).toThrow(login ? /expired/ : /No Galaxy credentials/);
      }
    });
    it("a profile key with no URL is refused, not sent to the stored host", () => {
      expect(() => run({ profile: "p" }, keys.profile)).toThrow(/has a key but no galaxy_url, and no Galaxy is named/);
    });
    it("a profile key with no URL and nothing stored says the profile has no galaxy_url", () => {
      const s = { env: {}, dotenv: {}, profiles: { p: { galaxy_user_key: "PROFKEY" } } };
      expect(() => resolveConnection({ profile: "p" }, s)).toThrow(/--profile p has a key but no galaxy_url/);
    });
    it("GALAXY_API_KEY and --api-key with no URL are refused too", () => {
      expect(() => run({}, keys.env)).toThrow(/No Galaxy credentials/);
      expect(() => run({ apiKey: "FLAG" })).toThrow(/No Galaxy credentials/);
    });
    it("an expired login for the only host cannot lend its host to a .env key", () => {
      expect(() => asExpired({}, keys.dotenv)).toThrow(expect.objectContaining({ kind: "not_logged_in" }));
    });
    it("an expired login hands over to a .env key for a host that was named", () => {
      const conn = asExpired({}, { dotenv: { GALAXY_URL: url, GALAXY_API_KEY: "DOTKEY" } });
      expect(conn).toMatchObject({ apiKey: "DOTKEY", baseUrl: url });
      expect(conn.warning).toMatch(/expired/);
    });
  });

  // One order, command line before environment before files (config.ts): --api-key, --profile,
  // GALAXY_API_KEY, the stored login, .env. Each adjacent pair is shown with the winner and the loser both
  // present, and every source is shown to be named in the result.
  describe("one order: --api-key, --profile, GALAXY_API_KEY, stored login, .env", () => {
    const profile = { profiles: { p: { galaxy_url: url, galaxy_user_key: "PROFKEY" } } };
    const named = { dotenv: { GALAXY_URL: url } };
    const all = [profile, named, keys.dotenv, keys.env];
    const who = (flags: GlobalFlags, ...parts: Partial<Sources>[]) => {
      const c = run(flags, ...parts);
      return [c.apiKey, c.keySource];
    };
    it("--api-key beats --profile", () => {
      expect(who({ profile: "p", apiKey: "FLAG" }, ...all)).toEqual(["FLAG", "flag"]);
    });
    it("--profile beats GALAXY_API_KEY (it ranked below it before stored logins existed)", () => {
      expect(who({ profile: "p" }, ...all)).toEqual(["PROFKEY", "profile"]);
    });
    it("GALAXY_API_KEY beats the stored login", () => {
      expect(who({}, named, keys.dotenv, keys.env)).toEqual(["ENVKEY", "env"]);
    });
    it("the stored login beats the .env key", () => {
      expect(who({}, named, keys.dotenv)).toEqual(["STORED", "stored"]);
    });
    it("the .env key is last: used when nothing else has a key for it", () => {
      const conn = asExpired({}, named, keys.dotenv);
      expect([conn.apiKey, conn.keySource]).toEqual(["DOTKEY", "dotenv"]);
    });
    // The two places where this differs from before stored logins, when the order was flag, env, .env, profile.
    it("--profile also beats a .env key (it ranked below it before stored logins existed)", () => {
      expect(who({ profile: "p" }, profile, named, keys.dotenv)).toEqual(["PROFKEY", "profile"]);
    });
    it("--profile with a URL only does not hide the stored login or the other keys", () => {
      const urlOnly = { profiles: { p: { galaxy_url: url } } };
      expect(who({ profile: "p" }, urlOnly)).toEqual(["STORED", "stored"]);
      expect(who({ profile: "p" }, urlOnly, keys.env)).toEqual(["ENVKEY", "env"]);
    });
    it("a profile key is never sent to the stored login's host", () => {
      expect(() => run({ profile: "p" }, keys.profile)).toThrow(/has a key but no galaxy_url, and no Galaxy is named/);
      expect(run({ profile: "p", url: "https://flag.example" }, keys.profile)).toMatchObject({ baseUrl: "https://flag.example", apiKey: "PROFKEY" });
    });
  });

  // A `.env` comes with the directory you are in, often someone else's clone: it is said out loud when it
  // chose the key or the Galaxy, and when a login of yours outranked its key.
  describe("saying where a .env was used", () => {
    const dir = { dotenvDir: "/work/repo" };
    it("names the file's directory when its key is used", () => {
      const conn = asExpired({}, dir, { dotenv: { GALAXY_URL: url, GALAXY_API_KEY: "DOTKEY" } });
      expect(conn.note).toBe("Using GALAXY_API_KEY and GALAXY_URL from /work/repo/.env.");
    });
    it("names it when only its GALAXY_URL chose the Galaxy, and says nothing of a key it did not use", () => {
      const conn = run({}, dir, { dotenv: { GALAXY_URL: url } });
      expect(conn.note).toBe("Using GALAXY_URL from /work/repo/.env.");
    });
    it("says when the stored login outranked its key, consistent with the above", () => {
      const conn = run({}, dir, keys.dotenv);
      expect(conn.note).toBe(`The stored login for ${url} ranks above GALAXY_API_KEY in /work/repo/.env, so that key is not used.`);
      const both = run({}, dir, { dotenv: { GALAXY_URL: url, GALAXY_API_KEY: "DOTKEY" } });
      expect(both.note).toBe(`Using GALAXY_URL from /work/repo/.env.\nThe stored login for ${url} ranks above GALAXY_API_KEY in /work/repo/.env, so that key is not used.`);
    });
    it("says nothing when no .env was involved, or a higher source overrode it", () => {
      expect(run({}, dir).note).toBeUndefined();
      expect(run({ url, apiKey: "FLAG" }, dir, { dotenv: { GALAXY_URL: "https://dot.example" } }).note).toBeUndefined();
    });
  });

  it("hints with the name it is given, gxy by default", () => {
    const s: Sources = { ...merge(), storedBaseUrls: [url], storedLogin: stored.expired };
    expect(() => resolveConnection({}, s)).toThrow(/run gxy auth login --host/);
    expect(() => resolveConnection({}, s, "galaxy-cli")).toThrow(/run galaxy-cli auth login --host/);
    expect(() => resolveConnection({}, merge(), "galaxy-cli")).toThrow(/or run galaxy-cli auth login\./);
  });
});

// A Galaxy that is named, with logins stored for others only: the error says that, and where to log in.
describe("the error when no credentials fit the Galaxy named", () => {
  const A = "https://a.example.org";
  const B = "https://b.example.org";
  const src = (stored: string[], over: Partial<Sources> = {}): Sources =>
    ({ env: {}, dotenv: {}, profiles: {}, storedBaseUrls: stored.map((u) => normalizeBaseUrl(u)), ...over });
  const message = (flags: GlobalFlags, s: Sources, name?: string) => {
    try {
      resolveConnection(flags, s, name);
    } catch (e) {
      return (e as Error).message;
    }
    throw new Error("expected resolveConnection to throw");
  };

  it("says the stored login is for another Galaxy, and how to log in to this one", () => {
    expect(message({ url: B }, src([A]))).toBe(
      `No Galaxy credentials for ${B}: the stored login is for ${A}. Run gxy auth login --host ${B}, or set GALAXY_API_KEY (env, a .env file, or --api-key) for it.`,
    );
  });
  it("lists every stored login when there are several, and uses the name the person typed", () => {
    expect(message({ url: B }, src([A, "https://c.example.org"]), "galaxy-cli")).toMatch(
      /^No Galaxy credentials for https:\/\/b\.example\.org: the stored logins are for https:\/\/a\.example\.org, https:\/\/c\.example\.org\. Run galaxy-cli auth login --host https:\/\/b\.example\.org,/,
    );
  });
  it("names the Galaxy a .env or GALAXY_URL chose just the same", () => {
    expect(message({}, src([A], { dotenv: { GALAXY_URL: B } }))).toContain(`credentials for ${B}:`);
    expect(message({}, src([A], { env: { GALAXY_URL: B } }))).toContain(`credentials for ${B}:`);
  });
  it("keeps the generic advice when nothing is named or nothing is stored", () => {
    expect(message({}, src([]))).toMatch(/^No Galaxy credentials\. Set GALAXY_URL and GALAXY_API_KEY/);
    expect(message({ url: B }, src([]))).toMatch(/^No Galaxy credentials\. Set GALAXY_URL/);
  });
  // Here a login for the Galaxy exists and could not be used, which is not "logged in to another Galaxy".
  it("keeps the generic advice, and the store's problem, when the login for the Galaxy named is there but unusable", () => {
    const m = message({ url: B }, src([B], { storeProblem: "the token cannot be read" }));
    expect(m).toMatch(/^No Galaxy credentials\. Set GALAXY_URL/);
    expect(m).toContain("(the token cannot be read)");
  });
  // The URL is shown without a user's credentials or query: the message may be pasted into an issue.
  it("shows the Galaxy without credentials or a query, and one that is not normalisable by its host", () => {
    const m = message({ url: "https://user:secret@b.example.org/galaxy?token=secret" }, src([A]));
    expect(m).not.toContain("secret");
    expect(m).toContain("credentials for b.example.org:");
    expect(message({ url: "http://remote.example/x" }, src([A]))).toContain("credentials for remote.example:");
  });
});

describe("loadSourcesAsync", () => {
  const url = normalizeBaseUrl("https://galaxy.example.org");
  const base = (over: Partial<Sources> = {}): Sources => ({ env: {}, dotenv: {}, profiles: {}, ...over });
  const seeded = () => new FakeStore().seed(fakeLogin({ baseUrl: url, token: "STORED", method: "pasted", scopes: [], expiresAt: null }));

  it("loads the login for the only stored host", async () => {
    const s = await loadSourcesAsync({}, { store: seeded(), base: base() });
    expect(s.storedLogin?.token).toBe("STORED");
    expect(resolveConnection({}, s).apiKey).toBe("STORED");
  });
  // The store is only read when no explicit key is given, so a broken hosts.yml never
  // gets in the way of `--api-key` or GALAXY_API_KEY.
  it("does not touch the store when a key is explicit", async () => {
    const boom = seeded();
    boom.list = async () => { throw new Error("no"); };
    const withProfile = base({ profiles: { p: { galaxy_user_key: "PK" } } });
    for (const [flags, b] of [[{ apiKey: "K" }, base()], [{}, base({ env: { GALAXY_API_KEY: "K" } })], [{ profile: "p" }, withProfile]] as const) {
      const s = await loadSourcesAsync(flags, { store: boom, base: b });
      expect(s.storedLogin).toBeUndefined();
      expect(s.storeProblem).toBeUndefined();
    }
  });
  // A broken store is a warning for an operation, not a usage error: see the last case in exit.test.ts.
  it("turns store failures into a problem rather than throwing", async () => {
    const boom = seeded();
    boom.list = async () => { throw new Error("hosts.yml is not valid"); };
    expect((await loadSourcesAsync({}, { store: boom, base: base() })).storeProblem).toMatch(/not valid/);
  });
  // The list leaves out a login this version cannot read (a newer gxy wrote it). For the Galaxy that was
  // asked for, `get` says so, and the operation's error carries it rather than only "no credentials".
  it("reports a login it cannot read for the Galaxy named, and not for any other", async () => {
    const store = new FakeStore();
    store.get = async (u) => {
      if (u === url) throw new AuthError("store", "The login for https://galaxy.example.org was written by a newer version of gxy.");
      return null;
    };
    const s = await loadSourcesAsync({}, { store, base: base({ env: { GALAXY_URL: url } }) });
    expect(s.storedLogin).toBeUndefined();
    expect(s.storeProblem).toContain("written by a newer version of gxy");
    expect(() => resolveConnection({}, s)).toThrow(/No Galaxy credentials.*written by a newer version of gxy/);
    const other = await loadSourcesAsync({}, { store, base: base({ env: { GALAXY_URL: "https://other.example.org" } }) });
    expect(other.storeProblem).toBeUndefined();
  });
  it("leaves an unparsable URL with no stored login, which is not an error here", async () => {
    const s = await loadSourcesAsync({}, { store: seeded(), base: base({ env: { GALAXY_URL: "not a url" } }) });
    expect(s.storedLogin).toBeUndefined();
    expect(s.storeProblem).toBeUndefined();
  });
  it("opens the default store from the config dir named in the environment", async () => {
    const dir = join(tmp.path, "cfg");
    await mkdir(dir, { mode: 0o700 });
    await writeFile(join(dir, "hosts.yml"), `hosts:\n  ${url}:\n    username: a\n    method: pasted\n    scopes: []\n    expires_at: null\n    obtained_at: "${DAY}"\n    storage: file\n    token: ${FAKE_TOKEN}\n`, { mode: 0o600 });
    const s = await loadSourcesAsync({}, { base: base({ env: { GXY_CONFIG_DIR: dir } }) });
    expect(s.storedLogin?.token).toBe(FAKE_TOKEN);
  });

  // Only a host that matches a stored login exactly gets its token. A `.env` that names another Galaxy
  // finds no login: the store is asked about that Galaxy, answers null, and is never asked for the stored one.
  describe("a stored token goes only to the host it was stored for", () => {
    const now = new Date("2026-06-01");
    it("a .env naming another Galaxy never gets it", async () => {
      const store = seeded();
      const get = vi.spyOn(store, "get");
      const b = base({ dotenv: { GALAXY_URL: "https://evil.example" }, now });
      const src = await loadSourcesAsync({}, { store, base: b, now });
      expect(get).not.toHaveBeenCalledWith(url);
      expect(src.storedLogin).toBeUndefined();
      expect(() => resolveConnection({}, src)).toThrow(/No Galaxy credentials/);
      // With a .env key too: that key goes to the .env host, the stored token still does not.
      expect(resolveConnection({}, { ...src, dotenv: { ...src.dotenv, GALAXY_API_KEY: "DOTKEY" } })).toMatchObject({ baseUrl: "https://evil.example", apiKey: "DOTKEY" });
    });
    // The whole path, from the store: a `.env` key with no URL, and an expired login
    // for the one stored Galaxy, must not send the key there.
    it("a .env key with no URL is not sent to the only stored Galaxy when its login cannot be used", async () => {
      const b = base({ dotenv: { GALAXY_API_KEY: "DOTKEY" }, now });
      const expired = new FakeStore().seed(fakeLogin({ baseUrl: url, token: "STORED", expiresAt: "2026-05-01T00:00:00.000Z" }));
      const fromExpired = await loadSourcesAsync({}, { store: expired, base: b, now });
      expect(() => resolveConnection({}, fromExpired)).toThrow(/expired/);
    });
    it("a .env naming the same Galaxy, however spelled, uses it", async () => {
      const b = base({ dotenv: { GALAXY_URL: "https://GALAXY.example.org/" }, now });
      expect(resolveConnection({}, await loadSourcesAsync({}, { store: seeded(), base: b, now }))).toMatchObject({ baseUrl: url, apiKey: "STORED" });
    });
  });
});

describe("loadSources (the files and environment)", () => {
  // Real files in a temp dir, with the working directory and HOME pointed at it: `loadSources` reads
  // `.env` from the working directory and `~/.planemo.yml` from HOME, and safe-env.ts leaves the
  // working directory alone.
  async function withFiles(files: Record<string, string>, fn: () => void) {
    for (const [name, text] of Object.entries(files)) await writeFile(join(tmp.path, name), text);
    const cwd = vi.spyOn(process, "cwd").mockReturnValue(tmp.path);
    vi.stubEnv("HOME", tmp.path);
    try {
      fn();
    } finally {
      cwd.mockRestore();
      vi.unstubAllEnvs();
    }
  }

  it("reads GALAXY_* from .env in the working directory, unquoted, ignoring other lines", async () => {
    await withFiles({ ".env": `# a comment\nGALAXY_URL="https://dot.example"\nGALAXY_API_KEY='DOTKEY'\nlower=ignored\n` }, () => {
      const s = loadSources();
      expect(s.dotenv).toMatchObject({ GALAXY_URL: "https://dot.example", GALAXY_API_KEY: "DOTKEY" });
      expect(s.dotenv.lower).toBeUndefined();
    });
  });
  it("reads planemo profiles from ~/.planemo.yml, and a top-level galaxy_url as the default one", async () => {
    await withFiles({ ".planemo.yml": "galaxy_url: https://def.example\ngalaxy_user_key: DEFKEY\nprofiles:\n  prod:\n    galaxy_url: https://prod.example\n    galaxy_user_key: PK\n" }, () => {
      expect(loadSources().profiles).toEqual({
        prod: { galaxy_url: "https://prod.example", galaxy_user_key: "PK" },
        default: { galaxy_url: "https://def.example", galaxy_user_key: "DEFKEY" },
      });
    });
  });
  it("has no .env and no profiles when the files are absent or unparsable", async () => {
    await withFiles({ ".planemo.yml": "{ unclosed" }, () => {
      const s = loadSources();
      expect(s.dotenv).toEqual({});
      expect(s.profiles).toEqual({});
    });
  });
});
