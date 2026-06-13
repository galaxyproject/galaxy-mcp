import createClient, { type Client, type ClientOptions } from "openapi-fetch";
import type { GalaxyApiPaths } from "./bindings";

export type GalaxyClient = Client<GalaxyApiPaths>;

/**
 * Our own client over the bindings -- explicit baseUrl + x-api-key.
 * Deliberately NOT the published createGalaxyApi (it pins openapi-fetch ^0.12 and
 * defaults baseUrl to window.location.origin). `fetchImpl` is for tests only.
 */
export function createGalaxyClient(
  baseUrl: string,
  apiKey: string,
  fetchImpl?: ClientOptions["fetch"],
): GalaxyClient {
  return createClient<GalaxyApiPaths>({
    baseUrl,
    headers: { "x-api-key": apiKey },
    ...(fetchImpl ? { fetch: fetchImpl } : {}),
  });
}
