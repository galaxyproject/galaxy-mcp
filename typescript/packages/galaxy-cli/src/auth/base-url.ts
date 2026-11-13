import { AuthError, type BaseUrl } from "./types";

/**
 * The one form a Galaxy's URL is stored and compared in.
 *
 * A bare host is taken as https. The path prefix is kept, because a Galaxy can be served
 * under one (`https://example.org/galaxy`) and two prefixes on one host are two Galaxies.
 * Query, fragment and trailing slashes are dropped, and `URL` itself lower-cases the host,
 * punycodes an IDN host and drops a default port, so `HOST.Example:443/` and `host.example`
 * are one key.
 *
 * Trailing `/api` segments are dropped too, however many: people paste the API root, and
 * `https://x/api` and `https://x` are the same Galaxy. Stripping them all keeps the result a
 * fixed point (`normalizeBaseUrl(normalizeBaseUrl(u)) === normalizeBaseUrl(u)`), which matters
 * because the result is a credential key and a stored key is normalised again when it is looked
 * up: with one `/api` stripped per pass, `.../api/api` would be stored under one key and looked
 * up under another. A segment that merely contains `api` (`/apiary`) is a different Galaxy and
 * stays.
 *
 * The key is exact: `https://x//galaxy` and `https://x/galaxy` are two keys, and `example.org.` (a
 * trailing dot) is not `example.org`. Nothing here guesses that they are the same Galaxy.
 *
 * A URL from anywhere is compared as `normalizeBaseUrl(url) === login.baseUrl`, so
 * `https://X/api/` finds the login stored for `x`.
 *
 * The Python server's `normalize_galaxy_url` refuses the same schemes, credentials, whitespace
 * and control characters; it refuses a query or fragment where this drops them, and has no
 * https default or `/api` stripping.
 *
 * Control characters are refused before anything is trimmed, so a stray `\r` from a CRLF file
 * or an ESC byte is an error rather than a quietly repaired key; only spaces around the URL
 * are trimmed, and whitespace inside it is refused. Comparison is
 * always exact, never by prefix. Credentials in the URL are refused because the URL is both
 * stored as the key and printed in messages, so `https://user:pw@host` would persist the
 * password and echo it. No error message prints the input for the same reason; they name only
 * the host or the scheme.
 *
 * Plain http is refused except to loopback: a token sent in the clear is a token handed
 * out to anyone on the path, and the only honest use for http here is a Galaxy on this
 * machine.
 */
export function normalizeBaseUrl(input: string): BaseUrl {
  if (/[\u0000-\u001f\u007f-\u009f]/.test(input))
    throw new AuthError("usage", "The Galaxy URL contains a control character.");
  const raw = input.trim();
  if (!raw) throw new AuthError("usage", "No Galaxy URL given.");
  if (/\s/.test(raw)) throw new AuthError("usage", "The Galaxy URL contains whitespace.");
  // `https:/x` (one slash) is a typo, not a bare host: left alone it would become `https://https/x`,
  // a different Galaxy with no error. A scheme-looking prefix is only taken as a host:port when
  // no slash follows the colon (`localhost:8080/galaxy`).
  if (/^[a-z][a-z0-9+.-]*:\/(?!\/)/i.test(raw))
    throw new AuthError("usage", "The Galaxy URL has a scheme but not its `//` (https://host).");
  const withScheme = /^[a-z][a-z0-9+.-]*:\/\//i.test(raw) ? raw : `https://${raw}`;
  let url: URL;
  try {
    url = new URL(withScheme);
  } catch {
    // Never `input`: it may carry a password (`https://user:secret@host:99999`).
    throw new AuthError("usage", "Not a URL: the Galaxy URL could not be parsed.");
  }
  // Credentials first, so a refused scheme cannot echo them either; after that no message below
  // prints more than the scheme or the host.
  if (url.username || url.password) {
    throw new AuthError("usage", `Leave credentials out of the Galaxy URL: ${url.host}`);
  }
  if (url.protocol !== "https:" && url.protocol !== "http:") {
    throw new AuthError(
      "usage",
      `Galaxy URLs are http or https, not ${url.protocol.replace(/:$/, "")}.`,
    );
  }
  if (url.protocol === "http:" && !isLoopback(url.hostname)) {
    throw new AuthError(
      "insecure",
      `Refusing plain http to ${url.host}: a token would cross the network unencrypted. Use https.`,
    );
  }
  let path = url.pathname.replace(/\/+$/, "");
  while (path.endsWith("/api")) path = path.slice(0, -"/api".length).replace(/\/+$/, "");
  return `${url.protocol}//${url.host}${path}` as BaseUrl;
}

/**
 * Whether `hostname` (as `URL` reports it, brackets and all) is this machine: `localhost`,
 * `::1`, or anything in `127.0.0.0/8`. Anchored on purpose, because a lookalike such as
 * `127.0.0.1.example.org` or `localhost.example.org` is somebody else's server and would
 * otherwise be trusted with a token in the clear. `0.0.0.0` is not loopback either.
 */
function isLoopback(hostname: string): boolean {
  const h = hostname.replace(/^\[|\]$/g, "");
  return h === "localhost" || h === "::1" || /^127(?:\.\d{1,3}){3}$/.test(h);
}
