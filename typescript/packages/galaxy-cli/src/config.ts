import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { parse as parseYaml } from "yaml";

export interface Connection { baseUrl: string; apiKey: string; }
export interface GlobalFlags { url?: string; apiKey?: string; profile?: string; }
interface PlanemoProfile { galaxy_url?: string; galaxy_user_key?: string; galaxy_admin_key?: string; }
export interface Sources {
  env: Record<string, string | undefined>;
  dotenv: Record<string, string | undefined>;
  profiles: Record<string, PlanemoProfile>;
  /** Where `dotenv` was read from, for messages. */
  dotenvPath?: string;
}

/** A key from outside .env was about to go to a Galaxy that the .env chose. Exits EX_NOPERM. */
export class CredentialScopeError extends Error {}

/**
 * Pure resolver -- sources injected so it is fully testable.
 *
 * A GALAXY_URL from .env is only used with the GALAXY_API_KEY from that same .env. A .env comes
 * with the directory, often a cloned repository, so it must not choose where a key from anywhere
 * else (--api-key, GALAXY_API_KEY in the environment, a profile) is sent. The rule depends only on
 * where the URL came from, not on how the sources are ranked.
 */
export function resolveConnection(flags: GlobalFlags, src: Sources): Connection {
  const prof = flags.profile ? src.profiles[flags.profile] : undefined;
  // Same order as the `??` chains below, so the label always names the value that wins.
  const hostFrom =
    flags.url !== undefined ? "flag"
    : src.env.GALAXY_URL !== undefined ? "env"
    : src.dotenv.GALAXY_URL !== undefined ? "dotenv"
    : prof?.galaxy_url !== undefined ? "profile"
    : undefined;
  const baseUrl = flags.url ?? src.env.GALAXY_URL ?? src.dotenv.GALAXY_URL ?? prof?.galaxy_url;
  const keyFrom =
    flags.apiKey !== undefined ? "flag"
    : src.env.GALAXY_API_KEY !== undefined ? "env"
    : src.dotenv.GALAXY_API_KEY !== undefined ? "dotenv"
    : (prof?.galaxy_user_key ?? prof?.galaxy_admin_key) !== undefined ? "profile"
    : undefined;
  const apiKey =
    flags.apiKey ?? src.env.GALAXY_API_KEY ?? src.dotenv.GALAXY_API_KEY ?? prof?.galaxy_user_key ?? prof?.galaxy_admin_key;
  if (!baseUrl || !apiKey) {
    throw new Error(
      "No Galaxy credentials. Set GALAXY_URL and GALAXY_API_KEY (env, a .env file, or --url/--api-key), " +
        "or pass --profile <name> for a ~/.planemo.yml profile.",
    );
  }
  if (hostFrom === "dotenv" && keyFrom !== "dotenv") {
    const dotenv = src.dotenvPath ?? ".env";
    const key =
      keyFrom === "flag" ? "--api-key"
      : keyFrom === "env" ? "GALAXY_API_KEY from the environment"
      : `the key of profile "${flags.profile}"`;
    throw new CredentialScopeError(
      `GALAXY_URL from ${dotenv} is only used with the GALAXY_API_KEY in the same file, not with ${key}. ` +
        "Set GALAXY_URL in the environment or pass --url to send that key to this Galaxy.",
    );
  }
  return { baseUrl, apiKey };
}

/** Read the real sources from disk/env (kept separate from the pure resolver). */
export function loadSources(): Sources {
  return { env: process.env, dotenv: readDotenv(), profiles: readPlanemoProfiles(), dotenvPath: join(process.cwd(), ".env") };
}

function readDotenv(): Record<string, string | undefined> {
  try {
    const text = readFileSync(join(process.cwd(), ".env"), "utf8");
    const out: Record<string, string> = {};
    for (const line of text.split("\n")) {
      const m = line.match(/^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*)\s*$/);
      if (m?.[1] !== undefined && m[2] !== undefined) out[m[1]] = m[2].replace(/^["']|["']$/g, "");
    }
    return out;
  } catch {
    return {};
  }
}

function readPlanemoProfiles(): Record<string, PlanemoProfile> {
  try {
    const doc = parseYaml(readFileSync(join(homedir(), ".planemo.yml"), "utf8")) as Record<string, unknown>;
    // planemo stores named profiles under `profiles:`; also accept top-level galaxy_url/key as "default".
    const profiles = (doc?.profiles as Record<string, PlanemoProfile>) ?? {};
    if (doc?.galaxy_url) profiles.default = { galaxy_url: String(doc.galaxy_url), galaxy_user_key: doc.galaxy_user_key as string | undefined };
    return profiles;
  } catch {
    return {};
  }
}
