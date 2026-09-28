import type { GalaxyContext } from "./context";
import { httpError } from "./errors";

/**
 * Narrow escape for Galaxy endpoints that are NOT in the OpenAPI bindings
 * (the classic tool controller). Reuses the already-configured client (baseUrl,
 * x-api-key, test fetchImpl) for an off-schema path. The cast is the only untyped
 * surface; the caller hand-writes T and MUST back it with a gated live test.
 */
type Verb = (
  p: string,
  i?: unknown,
) => Promise<{ data?: unknown; error?: unknown; response: { status: number } }>;

export async function legacyGet<T>(ctx: GalaxyContext, path: string, init?: unknown): Promise<T> {
  const get = ctx.client.GET as unknown as Verb;
  const { data, error, response } = await get(path, init);
  if (error || data == null) throw httpError(response, error);
  return data as T;
}

/** POST variant of {@link legacyGet} -- for off-schema endpoints like /api/unprivileged_tools. */
export async function legacyPost<T>(ctx: GalaxyContext, path: string, init?: unknown): Promise<T> {
  const post = ctx.client.POST as unknown as Verb;
  const { data, error, response } = await post(path, init);
  if (error || data == null) throw httpError(response, error);
  return data as T;
}

/** DELETE variant of {@link legacyGet} -- for off-schema deletes like /api/unprivileged_tools/{uuid}. */
export async function legacyDelete<T>(ctx: GalaxyContext, path: string, init?: unknown): Promise<T> {
  const del = ctx.client.DELETE as unknown as Verb;
  const { data, error, response } = await del(path, init);
  // DELETE may answer 204 No Content -- a null body on a 2xx is success, not an error.
  if (error) throw httpError(response, error);
  return data as T;
}
