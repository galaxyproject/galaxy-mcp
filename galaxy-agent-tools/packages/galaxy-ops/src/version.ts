// What /api/version says, and the whole of the comparison an op requirement needs.
//
// version_major is the only field worth comparing. Galaxy sets it to "YY.M" -- "26.0",
// "26.1", and "26.2" on a dev checkout -- and keeps the patch level in version_minor,
// which on that dev checkout is the string "dev0". Two integers is the entire ordering,
// so there is no semver dependency here and nothing to keep in step with one.

/** A version a server reported, kept with the text it reported so a refusal can quote it. */
export interface GalaxyVersion {
  /** What the server said, trimmed -- a refusal quotes this rather than the parsed pair. */
  readonly raw: string;
  readonly major: number;
  readonly minor: number;
}

/** The lower bound an op declares. */
export interface VersionRequirement {
  readonly major: number;
  readonly minor: number;
}

const OBSERVED = /^(\d+)\.(\d+)(?:[.\-+].*)?$/;
const REQUIREMENT = /^>=\s*(\d+)\.(\d+)$/;

/**
 * Read a version a server reported, or one a caller supplied.
 *
 * Lenient on the tail: a caller who passes the full "26.1.1" means the same server as
 * "26.1", and a dev checkout's "26.2" orders like any other. Anything that does not start
 * with two numbers is unknown rather than an error, because an unreadable version has to
 * refuse nothing.
 */
export function parseGalaxyVersion(raw: string | undefined | null): GalaxyVersion | undefined {
  if (typeof raw !== "string") return undefined;
  const m = OBSERVED.exec(raw.trim());
  if (!m) return undefined;
  return { raw: raw.trim(), major: Number(m[1]), minor: Number(m[2]) };
}

/**
 * Read an op's `requires.galaxy`. The grammar is `>=MAJOR.MINOR` and nothing else: that is
 * the shape version_major comes in, so a third component could never be answered, and no op
 * has yet needed an upper bound. A malformed one throws where the op registers rather than
 * at the call it would have been asked to judge.
 */
export function parseRequirement(spec: string): VersionRequirement {
  const m = REQUIREMENT.exec(spec.trim());
  if (!m) {
    throw new Error(`galaxy requirement ${JSON.stringify(spec)} is not of the form ">=MAJOR.MINOR"`);
  }
  return { major: Number(m[1]), minor: Number(m[2]) };
}

/** Whether a known server version meets a requirement. An unknown one is the caller's call. */
export function satisfiesRequirement(version: GalaxyVersion, spec: string): boolean {
  const want = parseRequirement(spec);
  if (version.major !== want.major) return version.major > want.major;
  return version.minor >= want.minor;
}

/** The sentence every surface appends, so no op writes its own and they cannot drift. */
export function requirementSentence(spec: string): string {
  const want = parseRequirement(spec);
  return `Requires Galaxy ${want.major}.${want.minor} or newer.`;
}
