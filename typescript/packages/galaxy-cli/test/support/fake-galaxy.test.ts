/**
 * The fake's own behaviour, so the tests that rely on it can rely on it. The whoami tests rely on
 * it refusing an unknown key the way Galaxy does. A fake that stopped answering 200 for a stranger
 * would let those tests pass without proving anything.
 */
import { describe, expect, it } from "vitest";
import { useFakeGalaxies } from "./fake-galaxy";

const start = useFakeGalaxies();
const userFor = async (baseUrl: string, key: string) => {
  const res = await fetch(`${baseUrl}/api/users/current`, { headers: { "x-api-key": key } });
  return { status: res.status, body: (await res.json()) as Record<string, unknown> };
};

describe("fake Galaxy", () => {
  it("knows a user by an issued token, and answers an unknown key as Galaxy 26.1 does: 200, no username", async () => {
    const galaxy = await start();
    const token = galaxy.issueToken({ admin: true, username: "root" });
    expect(await userFor(galaxy.baseUrl, token)).toMatchObject({
      status: 200,
      body: { username: "root", is_admin: true },
    });
    const anonymous = await userFor(galaxy.baseUrl, "fake-unknown");
    expect(anonymous.status).toBe(200);
    expect(anonymous.body).not.toHaveProperty("username");
  });

  it("can answer an unknown key with 403 instead, and a version probe with JSON", async () => {
    const galaxy = await start();
    galaxy.setInvalidKeyResponse("forbidden");
    expect((await userFor(galaxy.baseUrl, "fake-unknown")).status).toBe(403);
    const version = (await (await fetch(`${galaxy.baseUrl}/api/version`)).json()) as Record<
      string,
      unknown
    >;
    expect(version).toHaveProperty("version_major");
  });
});
