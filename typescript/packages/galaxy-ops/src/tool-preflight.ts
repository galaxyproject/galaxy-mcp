/**
 * What the Python server does before it submits a tool run, and says afterwards.
 *
 * Two things happen there that used to happen nowhere here. It reads the tool's schema and
 * checks the supplied inputs against it, and when it could not make that check it says so in
 * the success message -- `(inputs not pre-checked: <why>)` -- so a caller is never told
 * "started" in a way that implies the inputs were vetted when they were not. And it looks up
 * the caller's stored credentials for the tool and sends them with the run, saying
 * `(with credentials)` when it found some.
 *
 * The check itself needs Galaxy's parameter model walked -- conditionals, repeats, sections,
 * datatype compatibility -- which is a port of its own and is NOT here. What is here is
 * every part that decides which requests go out and which sentence comes back: when the
 * schema is read, the four reasons it can be unreadable, and the clause each one produces.
 * The consequence is stated plainly in the release notes: where that server would refuse a
 * run on its own evidence, this one submits it and lets Galaxy answer.
 *
 * server.py: `_supplies_a_reference`, `_get_tool_schema`, `_schema_to_check`,
 * `_preflight_tool_inputs`, `_get_tool_credentials_context`.
 */
import type { GalaxyContext } from "./context";
import { GalaxyError } from "./errors";
import { legacyGet } from "./legacy";
import { pyLibraryText } from "./python-failure";

/** A Galaxy `{"src": ..., "id": ...}` input value. `src` has to be a string to mean one. */
export function isReference(value: unknown): boolean {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value) &&
    typeof (value as { src?: unknown }).src === "string"
  );
}

/**
 * Whether any supplied value could be a dataset or collection reference.
 *
 * The checker only ever rejects a value `isReference` recognises, so a run made entirely of
 * scalars has nothing to check -- and saying so here is what skips an `io_details` fetch,
 * which builds the whole tool form server-side and scans the caller's history for every data
 * parameter. The same predicate the checker asks, so the two cannot drift apart.
 */
export function suppliesAReference(inputs: unknown): boolean {
  if (typeof inputs !== "object" || inputs === null || Array.isArray(inputs)) return false;
  const carriesSrc = (value: unknown): boolean => {
    if (Array.isArray(value)) return value.some(carriesSrc);
    if (typeof value === "object" && value !== null) {
      return isReference(value) || Object.values(value).some(carriesSrc);
    }
    return false;
  };
  return Object.values(inputs).some(carriesSrc);
}

/**
 * True when the schema actually carries an input list to check against.
 *
 * A tool with zero inputs is checkable and trivially fine; a schema fetched without
 * `io_details` has no `inputs` key at all and is not checkable. Reporting the second as
 * "checked, all clear" is the silent skip this exists to avoid.
 */
export function schemaHasInputs(schema: unknown): boolean {
  return (
    typeof schema === "object" &&
    schema !== null &&
    Array.isArray((schema as { inputs?: unknown }).inputs)
  );
}

/** True when this schema is for the exact tool that will be submitted. */
export function schemaDescribesTool(toolId: string, schema: unknown): boolean {
  if (typeof schema !== "object" || schema === null) return false;
  const id = (schema as { id?: unknown }).id;
  if (typeof id !== "string" || id === "") return false;
  // Galaxy expands an unversioned id to the installed version's full id.
  return id === toolId || id.startsWith(`${toolId}/`);
}

/** The schema to check inputs against, or the reason it cannot be checked on. */
export async function schemaToCheck(
  ctx: GalaxyContext,
  toolId: string,
  toolVersion?: string | null,
): Promise<{ schema: Record<string, unknown>; unchecked: string | null }> {
  let schema: Record<string, unknown>;
  try {
    schema = await legacyGet<Record<string, unknown>>(ctx, "/api/tools/{tool_id}", {
      params: {
        path: { tool_id: toolId },
        // The query bioblend's show_tool sends, plus the version when one was asked for --
        // Galaxy reads tool_version out of the query string here.
        query:
          toolVersion == null
            ? { io_details: true, link_details: false }
            : { io_details: true, link_details: false, tool_version: toolVersion },
      },
    });
  } catch (err) {
    // The reason quotes the exception, and the exception is the client library's: an
        // unversioned lookup goes through bioblend over there and a versioned one through
    // requests, which word a refusal differently.
    const facts = err instanceof GalaxyError ? err.http : undefined;
    const said = facts
      ? pyLibraryText(toolVersion == null ? "bioblend-get" : "raise-for-status", facts)
      : err instanceof Error
        ? err.message
        : String(err);
    return { schema: {}, unchecked: `could not fetch the schema for '${toolId}' (${said})` };
  }

  if (!schemaDescribesTool(toolId, schema)) {
    const described = schema["id"];
    return {
      schema,
      unchecked:
        `Galaxy returned a schema for '${described === undefined ? "None" : String(described)}', ` +
        `not '${toolId}', so it may describe a different version than the one being run`,
    };
  }

  if (toolVersion != null) {
    // Asking for a version is not the same as getting it: the toolbox returns the newest
    // installed version when the one asked for is missing. Checking one version's inputs
    // against another's parameters is how a preflight invents a mismatch.
    const served = schema["version"];
    if (typeof served === "string" && served !== toolVersion) {
      return {
        schema,
        unchecked:
          `Galaxy described version ${served} of '${toolId}' rather than the ` +
          `${toolVersion} asked for, so its parameters are not the ones this run would use`,
      };
    }
  }

  if (!schemaHasInputs(schema)) {
    return { schema, unchecked: `the definition of '${toolId}' arrived without a parameter list` };
  }

  return { schema, unchecked: null };
}

/**
 * Read the tool's schema before the run, and hand back why the inputs went unchecked.
 *
 * Null means there is nothing to say -- either the inputs carry no reference to check, or the
 * schema read fine. It never means "checked and clear" here, because the check itself is not
 * ported; see the module header.
 *
 * `schema` is for a caller that already holds the definition, as the user-tool run does: a
 * user-defined tool is scoped to its owner and never enters the toolbox, so looking it up by
 * id would just 404.
 */
export async function preflightToolInputs(
  ctx: GalaxyContext,
  toolId: string,
  inputs: unknown,
  opts: { schema?: Record<string, unknown> | null; toolVersion?: string | null } = {},
): Promise<string | null> {
  if (opts.schema != null) {
    return schemaHasInputs(opts.schema)
      ? null
      : `the definition of '${toolId}' arrived without a parameter list`;
  }
  if (!suppliesAReference(inputs)) return null;
  const { unchecked } = await schemaToCheck(ctx, toolId, opts.toolVersion);
  return unchecked;
}

/** One entry of the `credentials_context` a run carries. */
export interface CredentialContextEntry {
  user_credentials_id: unknown;
  name: unknown;
  version: unknown;
  selected_group: { id: unknown; name: unknown };
}

const own = (record: object, key: string): boolean =>
  Object.prototype.hasOwnProperty.call(record, key);

/**
 * The stored credentials for this tool, as the run payload carries them, or null.
 *
 * Two requests and a filter, and the filter is bioblend's rather than ours: a credential with
 * no current group, or one naming a group it does not list, is dropped, and an empty result is
 * null rather than an empty list -- which is what decides whether the run says
 * `(with credentials)`. Every failure answers null, because the other server wraps the whole
 * lookup in a suppress: a tool that needs no credentials must not fail for want of them.
 */
export async function toolCredentialsContext(
  ctx: GalaxyContext,
  toolId: string,
): Promise<CredentialContextEntry[] | null> {
  try {
    const user = await legacyGet<Record<string, unknown>>(ctx, "/api/users/{user_id}", {
      params: { path: { user_id: "current" } },
    });
    // `user_info["id"]` over there, which raises for a reply without one.
    if (!own(user, "id")) return null;
    const stored = await legacyGet<unknown>(ctx, "/api/users/{user_id}/credentials", {
      params: {
        path: { user_id: String(user["id"]) },
        query: { source_type: "tool", source_id: toolId },
      },
    });
    if (!Array.isArray(stored) || stored.length === 0) return null;
    const context: CredentialContextEntry[] = [];
    for (const entry of stored) {
      if (typeof entry !== "object" || entry === null) return null;
      const cred = entry as Record<string, unknown>;
      const groupId = own(cred, "current_group_id") ? cred["current_group_id"] : null;
      if (groupId === null || groupId === undefined) continue;
      const groups = cred["groups"];
      if (!Array.isArray(groups)) return null;
      const group = groups.find(
        (g) => typeof g === "object" && g !== null && (g as { id?: unknown }).id === groupId,
      ) as Record<string, unknown> | undefined;
      if (!group) continue;
      if (!own(cred, "id") || !own(cred, "name") || !own(cred, "version")) return null;
      context.push({
        user_credentials_id: cred["id"],
        name: cred["name"],
        version: cred["version"],
        selected_group: { id: groupId, name: group["name"] },
      });
    }
    return context.length > 0 ? context : null;
  } catch {
    return null;
  }
}
