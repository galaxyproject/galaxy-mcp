import createClient, { type Client, type ClientOptions } from "openapi-fetch";
import type { GalaxyApiPaths } from "./bindings";
import { transportFailure } from "./errors";
import { rememberHttpFailure } from "./http-failure";

export type GalaxyClient = Client<GalaxyApiPaths>;

/**
 * Our own client over the bindings -- explicit baseUrl + x-api-key.
 * Deliberately NOT the published createGalaxyApi (it pins openapi-fetch ^0.12 and
 * defaults baseUrl to window.location.origin). `fetchImpl` is for tests only.
 *
 * The middleware is what lets a failure be worded the way the Python server words one.
 * Its sentences quote the reply -- as bytes, as text, or by the URL asked for -- and none
 * of the three survives openapi-fetch's own handling: it parses the error body and hands
 * back the parsed value, `Response.url` is empty for a reply a test built, and the method
 * is on the request. So the facts are taken while the reply is still whole, and filed
 * against it for the throw site to pick up.
 */
export function createGalaxyClient(
  baseUrl: string,
  apiKey: string,
  fetchImpl?: ClientOptions["fetch"],
): GalaxyClient {
  const client = createClient<GalaxyApiPaths>({
    baseUrl,
    headers: { "x-api-key": apiKey },
    ...(fetchImpl ? { fetch: fetchImpl } : {}),
  });
  client.use({
    async onResponse({ request, response }) {
      // Only a failure needs this, and only a failure pays for the extra read: cloning a
      // reply that is about to be parsed anyway would double the work on every success.
      if (!response.ok) {
        rememberHttpFailure(response, {
          status: response.status,
          method: request.method.toUpperCase(),
          url: request.url,
          bodyText: await response.clone().text(),
          reason: response.statusText,
        });
      }
      return undefined;
    },
    onError({ request, error }) {
      // A request that never got a reply. The other server answers for this case too --
      // its client hands the tool a failure with no status on it -- so it is turned into
      // one of ours rather than left to escape as a bug.
      return transportFailure(request, error);
    },
  });
  return client;
}
