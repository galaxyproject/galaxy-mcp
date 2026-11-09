import { isLoopback } from "./base-url";

/**
 * The page a person is sent to approve a code, if it is somewhere they may safely be sent: https
 * (or http to this machine) on the Galaxy's own origin. Returns the URL as the WHATWG parser
 * normalises it, which is what must be printed and opened: the raw string can carry a tab or a
 * newline that the parser drops but a launcher or a terminal would not.
 *
 * The server picks this URL and we print and open it, so a Galaxy behind a misconfigured proxy
 * must not be able to point people at another host. RFC 8628 allows the page on another origin;
 * Galaxy's is its own, and a login that went elsewhere is stopped rather than trusted. The client
 * (`startDeviceLogin`) applies it, so a caller never sees a URL that is off the Galaxy.
 */
export function trustedVerificationUri(uri: string, baseUrl: string): string | undefined {
  try {
    const u = new URL(uri);
    if (u.origin !== new URL(baseUrl).origin) return undefined;
    const ok = u.protocol === "https:" || (u.protocol === "http:" && isLoopback(u.hostname));
    return ok ? u.href : undefined;
  } catch {
    return undefined;
  }
}
