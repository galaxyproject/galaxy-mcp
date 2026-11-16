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

/** The environment's key was about to go to a Galaxy it was not meant for. Exits EX_NOPERM. */
export class CredentialScopeError extends Error {}

/**
 * Pure resolver -- sources injected so it is fully testable.
 *
 * The environment's GALAXY_API_KEY is only sent to a Galaxy named by --url or by GALAXY_URL in the
 * environment, so a cloned repository's .env (or a profile) cannot point it somewhere else. A key
 * in .env still goes with the URL in that .env, a profile's key with its URL, and --api-key is
 * typed for the run, so those are unchanged.
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
  if (keyFrom === "env" && (hostFrom === "dotenv" || hostFrom === "profile")) {
    const origin = hostFrom === "dotenv" ? (src.dotenvPath ?? ".env") : `profile "${flags.profile}"`;
    throw new CredentialScopeError(
      `GALAXY_API_KEY is only sent to GALAXY_URL from the environment or --url; this Galaxy came from ${origin}. ` +
        "Set GALAXY_URL or pass --url to use it.",
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
