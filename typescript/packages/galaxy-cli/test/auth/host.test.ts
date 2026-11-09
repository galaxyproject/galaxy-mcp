/**
 * Which Galaxy is meant. `pickHost` is the one precedence rule, used by `gxy auth *` and by
 * operations alike, and `resolveHost` is its form for `auth`: it reads the stored logins only when
 * nothing else names a host, and normalises the result.
 */
import { describe, expect, it } from "vitest";
import { normalizeBaseUrl } from "../../src/auth/base-url";
import { pickHost, resolveHost } from "../../src/auth/host";
import { loadSourcesAsync, resolveConnection, type Sources } from "../../src/config";
import { FakeStore, fakeLogin } from "./harness";

const A = normalizeBaseUrl("https://a.example");
const B = normalizeBaseUrl("https://b.example");
const storeOf = (...urls: string[]) => {
  const s = new FakeStore();
  for (const u of urls) s.seed(fakeLogin({ baseUrl: normalizeBaseUrl(u) }));
  return s;
};
const stored = (...urls: string[]) => urls.map((u) => normalizeBaseUrl(u));
const empty = { env: {}, dotenv: {}, storedBaseUrls: [] as never[] };

describe("pickHost: one precedence", () => {
  const all = {
    flag: "https://flag.example",
    env: { GALAXY_URL: "https://env.example" },
    dotenv: { GALAXY_URL: "https://dot.example" },
    profileUrl: "https://prof.example",
    storedBaseUrls: stored("https://stored.example"),
  };
  // The command line beats the environment, which beats files; a profile is a pair, so its URL comes
  // right after the flag. Operations and every `auth` command but login use this one order.
  it("takes flag, then the profile, then env, then .env, then the only stored login", () => {
    expect(pickHost(all)).toEqual({ url: "https://flag.example", source: "flag" });
    expect(pickHost({ ...all, flag: undefined })).toEqual({
      url: "https://prof.example",
      source: "profile",
    });
    expect(pickHost({ ...all, flag: undefined, profileUrl: undefined }).source).toBe("env");
    expect(pickHost({ ...all, flag: undefined, profileUrl: undefined, env: {} }).source).toBe(
      "dotenv",
    );
    expect(
      pickHost({ ...all, flag: undefined, profileUrl: undefined, env: {}, dotenv: {} }),
    ).toEqual({
      url: "https://stored.example",
      source: "stored",
    });
  });

  it("for login, never takes the host from the profile or .env, whatever else is set", () => {
    expect(pickHost({ ...all, flag: undefined, env: {}, forLogin: true }).source).toBe("stored");
  });

  it("is ambiguous with several stored logins and none named, and a usage error with none", () => {
    const e = (() => {
      try {
        pickHost({ ...empty, storedBaseUrls: stored("https://a.example", "https://b.example") });
      } catch (x) {
        return x as Error;
      }
    })();
    expect(e).toMatchObject({ kind: "ambiguous_host" });
    expect(e?.message).toContain("https://a.example");
    expect(e?.message).toContain("https://b.example");
    expect(() => pickHost(empty)).toThrow(expect.objectContaining({ kind: "usage" }));
  });

  // `login` sends a key to the host it picks, so it never takes one from files
  // in the working or home directory: a repository must not be able to choose where your credentials go.
  it("for login, ignores .env and the profile but still takes the flag and the environment", () => {
    const l = { ...all, flag: undefined, env: {}, forLogin: true };
    expect(pickHost(l).source).toBe("stored");
    expect(() => pickHost({ ...l, storedBaseUrls: [] })).toThrow(
      expect.objectContaining({ kind: "usage" }),
    );
    expect(pickHost({ ...l, env: { GALAXY_URL: "https://env.example" } }).source).toBe("env");
    expect(pickHost({ ...l, flag: "https://flag.example" }).source).toBe("flag");
  });
});

describe("resolveHost (gxy auth *)", () => {
  it("normalises whatever won", async () => {
    expect(
      await resolveHost({ explicit: "Flag.example/", env: {}, dotenv: {}, store: storeOf() }),
    ).toBe("https://flag.example");
  });

  it("uses the profile URL, and the global --url as the flag", async () => {
    expect(
      await resolveHost({
        env: {},
        dotenv: {},
        profileUrl: "https://prof.example",
        store: storeOf("https://s.example"),
      }),
    ).toBe("https://prof.example");
    expect(
      await resolveHost({
        explicit: "Flag.example",
        env: {},
        dotenv: {},
        profileUrl: "https://prof.example",
        store: storeOf(),
      }),
    ).toBe("https://flag.example");
  });

  it("uses the only stored login, and fails when there are several or none", async () => {
    expect(await resolveHost({ env: {}, dotenv: {}, store: storeOf("https://s.example") })).toBe(
      "https://s.example",
    );
    await expect(
      resolveHost({
        env: {},
        dotenv: {},
        store: storeOf("https://a.example", "https://b.example"),
      }),
    ).rejects.toMatchObject({ kind: "ambiguous_host" });
    await expect(resolveHost({ env: {}, dotenv: {}, store: storeOf() })).rejects.toMatchObject({
      kind: "usage",
    });
  });

  it("refuses an insecure http host", async () => {
    await expect(
      resolveHost({ explicit: "http://remote.example", env: {}, dotenv: {}, store: storeOf() }),
    ).rejects.toMatchObject({ kind: "insecure" });
  });

  it("for login, never takes the host from .env or the profile", async () => {
    const input = {
      env: {},
      dotenv: { GALAXY_URL: "https://evil.example" },
      profileUrl: "https://evil.example",
      forLogin: true,
    };
    await expect(resolveHost({ ...input, store: storeOf() })).rejects.toMatchObject({
      kind: "usage",
    });
    expect(await resolveHost({ ...input, store: storeOf("https://s.example") })).toBe(
      "https://s.example",
    );
    expect(
      await resolveHost({ ...input, env: { GALAXY_URL: "https://env.example" }, store: storeOf() }),
    ).toBe("https://env.example");
  });

  // Listing the stored logins reads `hosts.yml`, so it happens only when nothing else names the host.
  it("does not list the store when something names the host", async () => {
    const store = storeOf("https://s.example");
    store.list = async () => {
      throw new Error("the store should not be read");
    };
    expect(
      await resolveHost({ explicit: "https://flag.example", env: {}, dotenv: {}, store }),
    ).toBe("https://flag.example");
  });
});

// Each case feeds the same inputs to the operations' own resolver (`loadSourcesAsync` and
// `resolveConnection`, config.ts) and to `auth`'s `resolveHost`, and compares what they pick. The
// expectation is the other code path's answer, not a restatement of the precedence, so the two cannot
// drift apart without this failing.
describe("operations and auth pick the same Galaxy", () => {
  const cases: Record<
    string,
    {
      flags: { url?: string; profile?: string };
      env?: Record<string, string>;
      dotenv?: Record<string, string>;
    }
  > = {
    flag: { flags: { url: B } },
    env: { flags: {}, env: { GALAXY_URL: B } },
    dotenv: { flags: {}, dotenv: { GALAXY_URL: B } },
    profile: { flags: { profile: "p" } },
    // The steps that used to differ: a profile URL against GALAXY_URL and a `.env` URL.
    "profile over env": { flags: { profile: "p" }, env: { GALAXY_URL: "https://env.example" } },
    "profile over .env": {
      flags: { profile: "p" },
      dotenv: { GALAXY_URL: "https://dot.example" },
    },
    "flag over profile": { flags: { url: "https://flag.example", profile: "p" } },
    stored: { flags: {} },
  };
  for (const [name, c] of Object.entries(cases)) {
    it(`agree on the ${name} step`, async () => {
      const store = storeOf(A);
      const profiles = { p: { galaxy_url: B, galaxy_user_key: "K" } };
      const base: Sources = {
        env: c.env ?? {},
        dotenv: { GALAXY_API_KEY: "K", ...c.dotenv },
        profiles,
        now: new Date("2026-06-01"),
      };
      const src = await loadSourcesAsync(c.flags, { store, base, now: base.now });
      const op = resolveConnection(c.flags, src).baseUrl;
      const auth = await resolveHost({
        explicit: c.flags.url,
        env: base.env,
        dotenv: base.dotenv,
        profileUrl: c.flags.profile ? B : undefined,
        store,
      });
      expect(normalizeBaseUrl(op)).toBe(auth);
    });
  }
});
