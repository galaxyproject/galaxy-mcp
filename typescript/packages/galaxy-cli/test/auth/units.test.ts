// Small pure modules of the auth surface, one section each.
import { describe, expect, it } from "vitest";
import { abortReason } from "../../src/auth/abort";
import { configDir } from "../../src/auth/paths";
import { normalizeBaseUrl } from "../../src/auth/base-url";
import { AuthError } from "../../src/auth/types";

// The order is the usual one for a CLI that follows the XDG Base Directory spec (gh does the
// same with its own variable). `env` and `home` are passed in, so no test reads the real ones.
describe("configDir", () => {
  it("prefers GXY_CONFIG_DIR, then XDG_CONFIG_HOME, then ~/.config", () => {
    expect(configDir({ GXY_CONFIG_DIR: "/x", XDG_CONFIG_HOME: "/xdg" }, "/home/u")).toBe("/x");
    expect(configDir({ XDG_CONFIG_HOME: "/xdg" }, "/home/u")).toBe("/xdg/gxy");
    expect(configDir({}, "/home/u")).toBe("/home/u/.config/gxy");
  });
  // The XDG spec says a relative path in `XDG_CONFIG_HOME` is invalid and must be ignored, and a
  // relative path would move the credential file with the working directory, possibly into a
  // repository. The same goes for `GXY_CONFIG_DIR`. An empty one counts as unset.
  it("ignores an empty GXY_CONFIG_DIR and a relative XDG_CONFIG_HOME", () => {
    expect(configDir({ GXY_CONFIG_DIR: "", XDG_CONFIG_HOME: "rel" }, "/home/u")).toBe(
      "/home/u/.config/gxy",
    );
  });
  it("ignores a relative GXY_CONFIG_DIR, falling through to the next in order", () => {
    expect(configDir({ GXY_CONFIG_DIR: "cfg" }, "/home/u")).toBe("/home/u/.config/gxy");
    expect(configDir({ GXY_CONFIG_DIR: "./cfg", XDG_CONFIG_HOME: "/xdg" }, "/home/u")).toBe(
      "/xdg/gxy",
    );
  });
});

// The result is the key every stored login is filed under, so these expectations are written out
// in full: a change to the normal form would silently re-key stored logins, and should fail here.
describe("normalizeBaseUrl", () => {
  it.each([
    ["Host.EXAMPLE", "https://host.example"],
    ["host.example:443", "https://host.example"],
    ["HOST.Example:443/", "https://host.example"],
    ["host.example/", "https://host.example"],
    ["host.example/api", "https://host.example"],
    ["host.example/api/", "https://host.example"],
    ["https://host.example/galaxy/api/", "https://host.example/galaxy"],
    ["host.example/galaxy", "https://host.example/galaxy"],
    ["https://host.example/galaxy//", "https://host.example/galaxy"],
    ["https://host.example/?x=1#frag", "https://host.example"],
    ["  https://host.example  ", "https://host.example"],
    ["http://127.0.0.1:8080/api", "http://127.0.0.1:8080"],
    ["https://bücher.example", "https://xn--bcher-kva.example"],
    ["https://host.example:8443", "https://host.example:8443"],
  ])("%s -> %s", (input, expected) => {
    expect(normalizeBaseUrl(input)).toBe(expected);
  });

  // The result is a credential key and `resolveHost` normalises a stored key again, so it must be a
  // fixed point. A doubled `/api` once normalised to a single `/api`, and that to the bare host, so
  // a login was looked up under a key that was never stored; a longer run of the idempotence
  // property in properties.test.ts found it, and the examples pin it here. The last row shows
  // that only trailing segments go: a leading `/api` is part of the path.
  it.each([
    ["https://h.example/api/api", "https://h.example"],
    ["https://h.example/api//api/", "https://h.example"],
    ["https://h.example/galaxy/api/api/api", "https://h.example/galaxy"],
    ["https://h.example/api/galaxy/api", "https://h.example/api/galaxy"],
  ])("strips every trailing /api, so normalising twice changes nothing: %s", (input, expected) => {
    const once = normalizeBaseUrl(input);
    expect(once).toBe(expected);
    expect(normalizeBaseUrl(once)).toBe(once);
  });

  // Two paths on one host are two Galaxies (a Galaxy can be served under a prefix), so only an
  // exact `/api` segment is the API root.
  it("keeps only an exact /api suffix: a longer segment is a different Galaxy", () => {
    expect(normalizeBaseUrl("https://host.example/apiary")).toBe("https://host.example/apiary");
    expect(normalizeBaseUrl("https://host.example/api/v2")).toBe("https://host.example/api/v2");
  });

  // Refused, not trimmed: a stray `\r` from a file with Windows line endings would otherwise become
  // a quietly different key. The inputs include U+0085 (a C1 control) and a space inside the host.
  it.each([
    "host.example\r",
    "host.example\n",
    "host.\u001bexample",
    "host.example\u0000",
    "ho st.example",
    "host.example/a b",
    "\u0085host.example",
  ])("refuses control characters and whitespace: %j", (input) => {
    expect(() => normalizeBaseUrl(input)).toThrow(AuthError);
  });

  // A password in the URL would be stored in `hosts.yml`, which is keyed by it.
  it("refuses userinfo, whatever the form", () => {
    expect(() => normalizeBaseUrl("https://user:pw@host.example")).toThrow(/credentials/);
    expect(() => normalizeBaseUrl("user@host.example")).toThrow(/credentials/);
  });

  // The input is never echoed: the credential check used to come after the parse and scheme
  // errors, which printed the whole string, password included.
  it.each([
    "https://user:SECRET@host.example:99999",
    "ftp://user:SECRET@host.example",
    "https://user:SECRET@",
    "user:SECRET@host.example:99999",
    "ftp://user:SECRET@host.example/path?token=SECRET",
  ])("never puts the password in an error: %s", (input) => {
    let message = "";
    try {
      normalizeBaseUrl(input);
    } catch (e) {
      message = (e as Error).message;
    }
    expect(message).not.toBe("");
    expect(message).not.toContain("SECRET");
  });

  // `https:/host` is a typo, and must not become the host "https" with path "/host".
  it.each(["https:/host.example", "http:/localhost", "HTTPS:/host.example/galaxy"])(
    "refuses a scheme with one slash: %s",
    (input) => {
      expect(() => normalizeBaseUrl(input)).toThrow(expect.objectContaining({ kind: "usage" }));
    },
  );

  it("still takes host:port, with or without a path, as a bare host", () => {
    expect(normalizeBaseUrl("host.example:8443")).toBe("https://host.example:8443");
    expect(normalizeBaseUrl("host.example:8443/galaxy")).toBe("https://host.example:8443/galaxy");
  });

  it("refuses plain http except to loopback, and other schemes", () => {
    expect(() => normalizeBaseUrl("http://host.example")).toThrow(
      expect.objectContaining({ kind: "insecure" }),
    );
    expect(() => normalizeBaseUrl("ftp://host.example")).toThrow(AuthError);
  });

  // The gate that decides whether a token may cross the network in clear text, so both sides are
  // pinned: what it accepts here, and (next) the hosts that only look like loopback. Dropping the
  // `$` of the 127.x pattern, for one, would send a token over http to
  // `127.0.0.1.evil.example`.
  it.each([
    ["http://localhost", "http://localhost"],
    ["http://localhost:8080", "http://localhost:8080"],
    ["http://[::1]:8080", "http://[::1]:8080"],
    ["http://127.0.0.1", "http://127.0.0.1"],
    ["http://127.1.2.3", "http://127.1.2.3"],
  ])("accepts plain http to loopback: %s", (input, expected) => {
    expect(normalizeBaseUrl(input)).toBe(expected);
  });

  // Each looks like loopback to a careless check and is not: a name an attacker could register, a
  // path that contains one, a neighbouring address (`[::2]`, and `0.0.0.0`, which is the
  // unspecified address), or userinfo in front of another host.
  it.each([
    ["http://127.0.0.1.evil.example", "insecure"],
    ["http://localhost.evil.example", "insecure"],
    ["http://evil.example/127.0.0.1", "insecure"],
    ["http://[::2]", "insecure"],
    ["http://0.0.0.0", "insecure"],
    ["http://127.0.0.1@evil.example", "usage"], // refused for the userinfo first, refused either way
  ])("refuses plain http to a host that only looks like loopback: %s", (input, kind) => {
    expect(() => normalizeBaseUrl(input)).toThrow(expect.objectContaining({ kind }));
  });
});

describe("abortReason", () => {
  it("passes an Error reason through as it is, so callers can still match on it", () => {
    const reason = new Error("ctrl-c");
    const controller = new AbortController();
    controller.abort(reason);
    expect(abortReason(controller.signal)).toBe(reason);
  });

  it("gives the AbortError of a plain abort()", () => {
    const controller = new AbortController();
    controller.abort();
    expect(abortReason(controller.signal)).toBeInstanceOf(Error);
    expect(abortReason(controller.signal).name).toBe("AbortError");
  });

  it("wraps a reason that is not an Error, keeping it as the cause", () => {
    const controller = new AbortController();
    controller.abort("stop");
    const error = abortReason(controller.signal);
    expect(error).toBeInstanceOf(Error);
    expect(error.cause).toBe("stop");
  });
});
