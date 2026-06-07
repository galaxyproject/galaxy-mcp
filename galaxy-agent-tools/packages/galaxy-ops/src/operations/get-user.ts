import type { GalaxyContext } from "../context";
import { classifyHttp, GalaxyAuthError } from "../errors";
import { register } from "./registry";
import type { AnyOperation, Operation } from "./types";

export interface CurrentUser {
  id: string;
  email: string;
  username: string;
}

const input = {}; // no args

async function run(_in: Record<string, never>, ctx: GalaxyContext): Promise<CurrentUser> {
  const { data, error, response } = await ctx.client.GET("/api/users/{user_id}", {
    params: { path: { user_id: "current" } },
  });
  if (error || !data) throw classifyHttp(response.status, error);
  // The endpoint returns DetailedUserModel | AnonUserModel; the anonymous shape has
  // no id/email/username. Surface that as an auth error rather than silently
  // returning undefined fields.
  const u = data as { id?: string; email?: string; username?: string };
  if (!u.id || !u.email || !u.username) {
    throw new GalaxyAuthError("Anonymous user response -- a valid API key is required");
  }
  return { id: u.id, email: u.email, username: u.username };
}

export const getUserOp: Operation<typeof input, CurrentUser> = {
  name: "get_user", // parity: mcp-server-galaxy-py get_user
  domain: "connection",
  summary: "Return the current authenticated Galaxy user (id, email, username).",
  input,
  run,
  project: (u) => ({ message: `Authenticated as ${u.username} <${u.email}>` }),
};

register(getUserOp as AnyOperation);

/** Code-mode entry. */
export const getUser = (i: Record<string, never>, ctx: GalaxyContext) => getUserOp.run(i, ctx);
