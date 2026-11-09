/**
 * The fake's own behaviour, so the tests that rely on it can rely on it. The end-to-end logout
 * asks the fake whether a token still works, and the login tests rely on it refusing an unknown
 * key the way Galaxy does. A fake that went on honouring a revoked token, or stopped answering
 * 200 for a stranger, would let those tests pass without proving anything.
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

  it("stops knowing a token once it is revoked or has expired", async () => {
    const galaxy = await start();
    const revoked = galaxy.issueToken();
    await fetch(`${galaxy.baseUrl}/oauth/revoke`, {
      method: "POST",
      headers: { "content-type": "application/x-www-form-urlencoded" },
      body: `token=${revoked}&client_id=galaxy-cli`,
    });
    expect(galaxy.isTokenValid(revoked)).toBe(false);
    expect((await userFor(galaxy.baseUrl, revoked)).body).not.toHaveProperty("username");

    const expired = galaxy.issueToken({ expiresIn: 0 });
    expect((await userFor(galaxy.baseUrl, expired)).body).not.toHaveProperty("username");
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

  // RFC 6749 §3.2.1 and RFC 7009 §2.1: a public client says who it is on every request, so a
  // client that forgot to would pass against a fake that did not check.
  it.each(["/oauth/device_authorization", "/oauth/token", "/oauth/revoke"])(
    "refuses %s without the client id, or with another one",
    async (path) => {
      const galaxy = await start();
      const post = async (body: string) =>
        (
          await fetch(`${galaxy.baseUrl}${path}`, {
            method: "POST",
            headers: { "content-type": "application/x-www-form-urlencoded" },
            body,
          })
        ).status;
      expect(await post("token=x&device_code=x&grant_type=x")).toBe(401);
      expect(await post("client_id=someone-else&token=x&device_code=x&grant_type=x")).toBe(401);
    },
  );

  it("refuses a revocation with no token", async () => {
    const galaxy = await start();
    const res = await fetch(`${galaxy.baseUrl}/oauth/revoke`, {
      method: "POST",
      headers: { "content-type": "application/x-www-form-urlencoded" },
      body: "client_id=galaxy-cli",
    });
    expect(res.status).toBe(400);
  });

  // The server's side of RFC 8628 §3.5, off by default: it is what lets a test see a client that
  // ignores `interval`.
  it("with enforceInterval, answers slow_down to a poll that comes too soon, and counts it", async () => {
    const galaxy = await start({ interval: 0.5, enforceInterval: true });
    const device = (await (
      await fetch(`${galaxy.baseUrl}/oauth/device_authorization`, {
        method: "POST",
        headers: { "content-type": "application/x-www-form-urlencoded" },
        body: "client_id=galaxy-cli",
      })
    ).json()) as { device_code: string };
    const poll = async () =>
      (
        (await (
          await fetch(`${galaxy.baseUrl}/oauth/token`, {
            method: "POST",
            headers: { "content-type": "application/x-www-form-urlencoded" },
            body: `client_id=galaxy-cli&grant_type=urn:ietf:params:oauth:grant-type:device_code&device_code=${device.device_code}`,
          })
        ).json()) as { error: string }
      ).error;
    expect(await poll()).toBe("authorization_pending");
    expect(await poll()).toBe("slow_down");
    expect(galaxy.tooFastPolls()).toBe(1);
  });
});
