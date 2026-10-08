import { describe, it, expect } from "vitest";
import { resolveConnection } from "../src/config";

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
