import { version } from "../../package.json";

/** What the auth code calls itself on the wire, so a Galaxy admin can tell `gxy` from a browser. */
export const USER_AGENT = `gxy/${version} (galaxy-cli)`;

/** `fetch` with our User-Agent on every request (openid-client's `customFetch` takes this shape). */
export function withUserAgent(inner: typeof fetch = globalThis.fetch): typeof fetch {
  return (input, init) => {
    const headers = new Headers(
      init?.headers ?? (input instanceof Request ? input.headers : undefined),
    );
    headers.set("user-agent", USER_AGENT);
    return inner(input, { ...init, headers });
  };
}
