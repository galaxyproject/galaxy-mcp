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
    const prod = { galaxy_url: "https://prof.example", galaxy_user_key: "PK" };
    // dotenv URL and key win over the profile's
    expect(
      resolveConnection(
        { profile: "prod" },
        { env: {}, dotenv: { GALAXY_URL: "https://dot.example", GALAXY_API_KEY: "DK" }, profiles: { prod } },
      ),
    ).toEqual({ baseUrl: "https://dot.example", apiKey: "DK" });
    // key only in profile, URL from env
    expect(
      resolveConnection({ profile: "prod" }, { env: { GALAXY_URL: "https://env.example" }, dotenv: {}, profiles: { prod } }),
    ).toEqual({ baseUrl: "https://env.example", apiKey: "PK" });
  });
  it("throws a clear error when no credentials resolve", () => {
    expect(() => resolveConnection({}, { env: {}, dotenv: {}, profiles: {} })).toThrow(/GALAXY_URL|GALAXY_API_KEY/);
  });
});

describe("resolveConnection: a .env URL only goes with the .env key", () => {
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

  // Refused exactly when the URL that wins came from .env and the key that wins did not.
  const hosts = ["flag", "env", "dotenv", "profile"] as const;
  const keys = ["flag", "env", "dotenv", "profile"] as const;
  for (const h of hosts) {
    for (const k of keys) {
      const refused = h === "dotenv" && k !== "dotenv";
      it(`host from ${h}, key from ${k}: ${refused ? "refused" : "sent"}`, () => {
        if (refused) expect(() => run(h, k)).toThrow(CredentialScopeError);
        else expect(run(h, k)).toEqual({ baseUrl: URL_OF[h], apiKey: KEY_OF[k] });
      });
    }
  }

  it("names the .env file, the key and how to proceed", () => {
    expect(() => run("dotenv", "env")).toThrow(
      "GALAXY_URL from /repo/.env is only used with the GALAXY_API_KEY in the same file, not with GALAXY_API_KEY from the environment. " +
        "Set GALAXY_URL in the environment or pass --url to send that key to this Galaxy.",
    );
  });
  it("names --api-key and a profile's key", () => {
    expect(() => run("dotenv", "flag")).toThrow("not with --api-key.");
    expect(() => run("dotenv", "profile")).toThrow('not with the key of profile "p".');
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
  it("does not send a profile's key to a .env URL that outranks the profile's own", () => {
    expect(() =>
      resolveConnection({ profile: "p" }, {
        env: {},
        dotenv: { GALAXY_URL: "https://dot.example" },
        profiles: { p: { galaxy_url: "https://prof.example", galaxy_user_key: "PK" } },
      }),
    ).toThrow(CredentialScopeError);
  });
  it("sends the environment's key to a profile's URL: the profile is the user's own file", () => {
    expect(
      resolveConnection({ profile: "p" }, {
        env: { GALAXY_API_KEY: "EK" },
        dotenv: {},
        profiles: { p: { galaxy_url: "https://prof.example", galaxy_user_key: "PK" } },
      }),
    ).toEqual({ baseUrl: "https://prof.example", apiKey: "EK" });
  });
  it("still reports missing credentials as a plain error", () => {
    expect(() => resolveConnection({}, { env: {}, dotenv: {}, profiles: {} })).not.toThrow(CredentialScopeError);
  });
});
