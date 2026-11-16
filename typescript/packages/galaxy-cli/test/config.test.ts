import { describe, it, expect, vi, afterEach } from "vitest";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { resolveConnection, loadSources } from "../src/config";

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

describe("loadSources: reading .env", () => {
  let dir: string | undefined;
  afterEach(() => {
    vi.restoreAllMocks();
    if (dir) rmSync(dir, { recursive: true, force: true });
    dir = undefined;
  });
  function dotenvOf(text: string) {
    dir = mkdtempSync(join(tmpdir(), "galaxy-cli-dotenv-"));
    writeFileSync(join(dir, ".env"), text);
    vi.spyOn(process, "cwd").mockReturnValue(dir);
    return loadSources().dotenv;
  }

  it("reads a file with CRLF line endings, without keeping the \\r", () => {
    expect(dotenvOf("GALAXY_URL=https://dot.example\r\nGALAXY_API_KEY=DK\r\n")).toEqual({
      GALAXY_URL: "https://dot.example",
      GALAXY_API_KEY: "DK",
    });
  });
  it("drops trailing whitespace before stripping quotes", () => {
    expect(dotenvOf('GALAXY_URL="https://dot.example"  \nGALAXY_API_KEY=\'DK\'\t\r\n')).toEqual({
      GALAXY_URL: "https://dot.example",
      GALAXY_API_KEY: "DK",
    });
  });
  it("keeps whitespace inside a value", () => {
    expect(dotenvOf("NAME=  a b  \n")).toEqual({ NAME: "a b" });
  });
});
