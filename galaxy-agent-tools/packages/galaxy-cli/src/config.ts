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
}

/** Pure resolver -- sources injected so it is fully testable. */
export function resolveConnection(flags: GlobalFlags, src: Sources): Connection {
  const prof = flags.profile ? src.profiles[flags.profile] : undefined;
  const baseUrl =
    flags.url ?? src.env.GALAXY_URL ?? src.dotenv.GALAXY_URL ?? prof?.galaxy_url;
  const apiKey =
    flags.apiKey ?? src.env.GALAXY_API_KEY ?? src.dotenv.GALAXY_API_KEY ?? prof?.galaxy_user_key ?? prof?.galaxy_admin_key;
  if (!baseUrl || !apiKey) {
    throw new Error(
      "No Galaxy credentials. Set GALAXY_URL and GALAXY_API_KEY (env, a .env file, or --url/--api-key), " +
        "or pass --profile <name> for a ~/.planemo.yml profile.",
    );
  }
  return { baseUrl, apiKey };
}

/** Read the real sources from disk/env (kept separate from the pure resolver). */
export function loadSources(): Sources {
  return { env: process.env, dotenv: readDotenv(), profiles: readPlanemoProfiles() };
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
