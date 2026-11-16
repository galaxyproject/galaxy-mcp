import { describe, it, expect } from "vitest";
import { resolveConnection, CredentialScopeError } from "../src/config";

describe("resolveConnection", () => {
  it("prefers explicit flags over env", () => {
    const conn = resolveConnection(
      { url: "https://flag.example", apiKey: "FLAGKEY" },
      { env: { GALAXY_URL: "https://env.example", GALAXY_API_KEY: "ENVKEY" }, dotenv: {}, profiles: {} },
    );
    expect(conn).toEqual({ baseUrl: "https://flag.example", apiKey: "FLAGKEY" });
  });
  it("falls back to env, then dotenv, then profile", () => {
    const conn = resolveConnection(
      { profile: "prod" },
      { env: {}, dotenv: { GALAXY_URL: "https://dot.example" }, profiles: { prod: { galaxy_url: "https://prof.example", galaxy_user_key: "PK" } } },
    );
    expect(conn.baseUrl).toBe("https://dot.example"); // dotenv URL wins over profile
    expect(conn.apiKey).toBe("PK"); // key only in profile
  });
  it("throws a clear error when no credentials resolve", () => {
    expect(() => resolveConnection({}, { env: {}, dotenv: {}, profiles: {} })).toThrow(/GALAXY_URL|GALAXY_API_KEY/);
  });
});

describe("resolveConnection: the environment's key and where the host came from", () => {
  const HOST = {
    flag: { flags: { url: "https://flag.example" }, env: {}, dotenv: {} },
    env: { flags: {}, env: { GALAXY_URL: "https://env.example" }, dotenv: {} },
    dotenv: { flags: {}, env: {}, dotenv: { GALAXY_URL: "https://dot.example" } },
    profile: { flags: { profile: "p" }, env: {}, dotenv: {} },
  } as const;
  const URL_OF = {
    flag: "https://flag.example", env: "https://env.example", dotenv: "https://dot.example", profile: "https://prof.example",
  } as const;
  type Key = "flag" | "env" | "dotenv" | "profile";
  const KEY = {
    flag: { flags: { apiKey: "FK" }, env: {}, dotenv: {}, profile: {} },
    env: { flags: {}, env: { GALAXY_API_KEY: "EK" }, dotenv: {}, profile: {} },
    dotenv: { flags: {}, env: {}, dotenv: { GALAXY_API_KEY: "DK" }, profile: {} },
    profile: { flags: { profile: "p" }, env: {}, dotenv: {}, profile: { galaxy_user_key: "PK" } },
  } as const;
  const KEY_OF = { flag: "FK", env: "EK", dotenv: "DK", profile: "PK" } as const;

  function run(h: keyof typeof HOST, k: Key) {
    const H = HOST[h], K = KEY[k];
    return resolveConnection(
      { ...H.flags, ...K.flags },
      {
        env: { ...H.env, ...K.env },
        dotenv: { ...H.dotenv, ...K.dotenv },
        profiles: { p: { galaxy_url: "https://prof.example", ...K.profile } },
        dotenvPath: "/repo/.env",
      },
    );
  }

  // Key precedence is flag > env > dotenv > profile, so the key that wins is the highest listed.
  const hosts = ["flag", "env", "dotenv", "profile"] as const;
  const keys = ["flag", "env", "dotenv", "profile"] as const;
  for (const h of hosts) {
    for (const k of keys) {
      const refused = k === "env" && (h === "dotenv" || h === "profile");
      it(`host from ${h}, key from ${k}: ${refused ? "refused" : "sent"}`, () => {
        if (refused) expect(() => run(h, k)).toThrow(CredentialScopeError);
        else expect(run(h, k)).toEqual({ baseUrl: URL_OF[h], apiKey: KEY_OF[k] });
      });
    }
  }

  it("names the .env file and how to proceed", () => {
    expect(() => run("dotenv", "env")).toThrow(
      "GALAXY_API_KEY is only sent to GALAXY_URL from the environment or --url; this Galaxy came from /repo/.env. " +
        "Set GALAXY_URL or pass --url to use it.",
    );
  });
  it("names the profile and how to proceed", () => {
    expect(() => run("profile", "env")).toThrow(
      'GALAXY_API_KEY is only sent to GALAXY_URL from the environment or --url; this Galaxy came from profile "p". ' +
        "Set GALAXY_URL or pass --url to use it.",
    );
  });
  it("refuses, rather than switching keys, when .env has URL and key and the environment has a key", () => {
    expect(() =>
      resolveConnection({}, {
        env: { GALAXY_API_KEY: "EK" },
        dotenv: { GALAXY_URL: "https://dot.example", GALAXY_API_KEY: "DK" },
        profiles: {},
      }),
    ).toThrow(CredentialScopeError);
  });
  it("refuses when a profile has its own key but the environment's key outranks it", () => {
    expect(() =>
      resolveConnection({ profile: "p" }, {
        env: { GALAXY_API_KEY: "EK" },
        dotenv: {},
        profiles: { p: { galaxy_url: "https://prof.example", galaxy_user_key: "PK" } },
      }),
    ).toThrow(CredentialScopeError);
  });
  it("still reports missing credentials as a plain error", () => {
    expect(() => resolveConnection({}, { env: {}, dotenv: {}, profiles: {} })).not.toThrow(CredentialScopeError);
  });
});
