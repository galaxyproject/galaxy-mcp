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

/**
 * Whether Galaxy answered with a failure, decided the one way that is always true.
 *
 * The status, and never the body: openapi-fetch reads a failed reply's body as text and
 * parses it if it can, so a reply with no body at all hands back the empty string -- which
 * is falsy. A guard asking "is there an error?" therefore answered "no" for a 403 with an
 * empty body, and `delete_user_tool` reported the tool deactivated. The other server reaches
 * the same conclusion the same way: `response.raise_for_status()` looks at the status line
 * and never at what came after it.
 *
 * Non-2xx rather than >=400, so this agrees with the client middleware that files the facts
 * a sentence is worded from: it files them for `!response.ok`, and a status that leaves no
 * facts behind must not be the one status that gets past the guard. A reply that is missing
 * altogether is a failure for the same reason -- nothing says the request succeeded.
 */
const answeredWithAFailure = (response: { status: number } | undefined): boolean =>
  !response || response.status < 200 || response.status >= 300;

export async function legacyGet<T>(ctx: GalaxyContext, path: string, init?: unknown): Promise<T> {
  const get = ctx.client.GET as unknown as Verb;
  const { data, error, response } = await get(path, init);
  if (answeredWithAFailure(response) || error || data == null) {
    throw httpError(response, error);
  }
  return data as T;
}

/** POST variant of {@link legacyGet} -- for off-schema endpoints like /api/unprivileged_tools. */
export async function legacyPost<T>(ctx: GalaxyContext, path: string, init?: unknown): Promise<T> {
  const post = ctx.client.POST as unknown as Verb;
  const { data, error, response } = await post(path, init);
  if (answeredWithAFailure(response) || error || data == null) {
    throw httpError(response, error);
  }
  return data as T;
}

/** DELETE variant of {@link legacyGet} -- for off-schema deletes like /api/unprivileged_tools/{uuid}. */
export async function legacyDelete<T>(ctx: GalaxyContext, path: string, init?: unknown): Promise<T> {
  const del = ctx.client.DELETE as unknown as Verb;
  const { data, error, response } = await del(path, init);
  // The status alone, because a DELETE may answer 204 No Content: a null body on a 2xx is
  // success, and the same null body on a 403 is not.
  if (answeredWithAFailure(response) || error) throw httpError(response, error);
  return data as T;
}
