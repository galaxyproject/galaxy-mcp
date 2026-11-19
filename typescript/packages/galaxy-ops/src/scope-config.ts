/**
 * A scope read from a set of named settings, for the two programs.
 *
 * The library reads no environment: `galaxy-mcp` and `galaxy-cli` hand their own
 * `process.env` to this and pass the result to `createGalaxyContext`. A setting that cannot
 * be read is an error with the variable's name in it, thrown before anything connects, because
 * a scope that silently fell back to "unrestricted" is the one failure this must not have.
 */
import type { OperationScope } from "./scope";

export const SCOPE_ENV = {
  write: "GALAXY_MCP_SCOPE_WRITE_HISTORIES",
  read: "GALAXY_MCP_SCOPE_READ_HISTORIES",
  createdOnly: "GALAXY_MCP_SCOPE_CREATED_ONLY",
  allowCreate: "GALAXY_MCP_SCOPE_ALLOW_CREATE",
  destructive: "GALAXY_MCP_SCOPE_DESTRUCTIVE",
  blocked: "GALAXY_MCP_SCOPE_BLOCKED_TOOLS",
} as const;

const ENCODED_ID = /^[0-9a-fA-F]+$/;

function items(value: string | undefined): string[] {
  return (value ?? "").split(",").map((v) => v.trim()).filter((v) => v !== "");
}

function flag(name: string, value: string | undefined): boolean | undefined {
  if (value === undefined || value.trim() === "") return undefined;
  const v = value.trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(v)) return true;
  if (["0", "false", "no", "off"].includes(v)) return false;
  throw new Error(`${name} must be 1 or 0 (got '${value}')`);
}

function historyIds(name: string, value: string | undefined): string[] {
  const ids = items(value);
  for (const id of ids) {
    if (!ENCODED_ID.test(id)) {
      throw new Error(`${name} must be a comma-separated list of encoded history ids; '${id}' is not one`);
    }
  }
  return ids;
}

/**
 * The scope these settings describe, or undefined when none of them is set -- which is an
 * unscoped program, exactly as before. `knownOperations` checks the blocked list for typos.
 */
export function scopeFromEnv(
  env: Readonly<Record<string, string | undefined>>,
  knownOperations?: readonly string[],
): OperationScope | undefined {
  const set = Object.values(SCOPE_ENV).filter((name) => (env[name] ?? "").trim() !== "");
  if (set.length === 0) return undefined;

  const write = historyIds(SCOPE_ENV.write, env[SCOPE_ENV.write]);
  const read = historyIds(SCOPE_ENV.read, env[SCOPE_ENV.read]);
  const createdOnly = flag(SCOPE_ENV.createdOnly, env[SCOPE_ENV.createdOnly]) === true;
  const allowCreate = flag(SCOPE_ENV.allowCreate, env[SCOPE_ENV.allowCreate]) !== false;
  const destructive = (env[SCOPE_ENV.destructive] ?? "").trim().toLowerCase() || "created";
  if (destructive !== "created" && destructive !== "writable") {
    throw new Error(`${SCOPE_ENV.destructive} must be 'created' or 'writable' (got '${env[SCOPE_ENV.destructive]}')`);
  }
  if (createdOnly && write.length > 0) {
    throw new Error(`${SCOPE_ENV.createdOnly}=1 means no pre-existing history may be written; unset ${SCOPE_ENV.write} or it`);
  }
  if (createdOnly && !allowCreate) {
    throw new Error(`${SCOPE_ENV.createdOnly}=1 with ${SCOPE_ENV.allowCreate}=0 leaves no history the session could ever write`);
  }
  if (!createdOnly && write.length === 0 && read.length === 0) {
    throw new Error(
      `a scope needs histories: set ${SCOPE_ENV.write} and/or ${SCOPE_ENV.read}, or ${SCOPE_ENV.createdOnly}=1 ` +
        `(only ${set.join(", ")} set)`,
    );
  }
  const blocked = items(env[SCOPE_ENV.blocked]);
  if (knownOperations) {
    const unknown = blocked.filter((name) => !knownOperations.includes(name));
    if (unknown.length > 0) throw new Error(`${SCOPE_ENV.blocked} names unknown tools: ${unknown.join(", ")}`);
  }
  return {
    historyIds: write,
    readOnlyHistoryIds: read,
    blockedOperations: blocked,
    includeCreated: true,
    allowCreate,
    destructive,
  };
}
