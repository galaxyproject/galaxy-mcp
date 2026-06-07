import type { GalaxyContext } from "./context";
import { classifyHttp } from "./errors";

/**
 * Narrow escape for Galaxy endpoints that are NOT in the OpenAPI bindings
 * (the classic tool controller). Reuses the already-configured client (baseUrl,
 * x-api-key, test fetchImpl) for an off-schema path. The cast is the only untyped
 * surface; the caller hand-writes T and MUST back it with a gated live test.
 */
export async function legacyGet<T>(ctx: GalaxyContext, path: string, init?: unknown): Promise<T> {
  const get = ctx.client.GET as unknown as (
    p: string,
    i?: unknown,
  ) => Promise<{ data?: unknown; error?: unknown; response: { status: number } }>;
  const { data, error, response } = await get(path, init);
  if (error || data == null) throw classifyHttp(response.status, error);
  return data as T;
}
