import { z } from "zod";
import type { GalaxyContext } from "../context";
import { GalaxyValidationError } from "../errors";
import {
  biocontainerTagBuilt,
  recommendContainer,
  type MatchQuality,
  type PackageSpec,
  type RecommendationSource,
} from "../mulled";
import { pyRepr, pyStrip } from "../python-str";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

export interface BiocontainerRecommendation {
  image: string | null;
  found: boolean;
  match_quality: MatchQuality;
  source: RecommendationSource;
  notes: string[];
  verified: boolean | null;
}

const input = {
  packages: z
    .array(z.string())
    .min(1, "packages must contain at least one conda package name")
    .describe(
      'The conda packages the tool wraps, each as "name" or\n' +
        '"name=version" (e.g. ["samtools=1.17", "bwa"]). Use canonical conda\n' +
        'names you would `conda install` (e.g. "pandas", "r-ggplot2",\n' +
        '"samtools"). A single package yields a single-package image; several\n' +
        "yield a mulled-v2 image.",
    ),
};
type In = { packages: string[] };

/** Python's `name` / `name=version` parsing, refusals included. */
export function parsePackageSpecs(packages: readonly string[]): PackageSpec[] {
  const parsed: PackageSpec[] = [];
  for (const pkg of packages) {
    const at = pkg.indexOf("=");
    // pyStrip, not trim: Python strips U+0085 and U+001C-U+001F and keeps U+FEFF, and
    // JavaScript does the opposite on both -- see the note on pyStrip in python-str.ts.
    const name = pyStrip(at === -1 ? pkg : pkg.slice(0, at));
    const version = at === -1 ? "" : pyStrip(pkg.slice(at + 1));
    if (!name) {
      // Python's message is `f"invalid package entry {pkg!r}: ..."`, so the entry arrives
      // escaped and quoted the way `repr` quotes it: a newline reads as `\n` rather than
      // breaking the line, and an entry holding a single quote is quoted with double ones.
      throw new GalaxyValidationError(
        `invalid package entry ${pyRepr(pkg)}: expected 'name' or 'name=version'`,
      );
    }
    parsed.push({ name, version: version || null });
  }
  if (parsed.length === 0) {
    throw new GalaxyValidationError("packages must contain at least one conda package name");
  }
  return parsed;
}

async function run(i: In, _ctx: GalaxyContext): Promise<BiocontainerRecommendation> {
  const specs = parsePackageSpecs(i.packages);
  const recommendation = await recommendContainer(specs);
  const verified = recommendation.image ? await biocontainerTagBuilt(recommendation.image) : null;
  return {
    image: recommendation.image,
    found: recommendation.found,
    match_quality: recommendation.match_quality,
    source: recommendation.source,
    notes: [...recommendation.notes],
    verified,
  };
}

export const recommendBiocontainerOp: Operation<typeof input, BiocontainerRecommendation> = {
  name: "recommend_biocontainer",
  domain: "tools",
  summary:
    "Resolve a verified quay.io/biocontainers image for a set of conda packages. " +
    "Use this to pick the `container` for create_user_tool instead of guessing an image. " +
    "The result is verified against quay.io rather than hallucinated, which avoids the most " +
    'common user-defined-tool failure: inventing a tag, or using a bare image (e.g. "python:3.12-slim") ' +
    "that doesn't ship the libraries the tool imports. " +
    "match_quality is exact_version | name_only | not_found, where name_only means the package " +
    "names matched but no exact-version combination was established -- either the pinned version " +
    "has no built tag, or no version was pinned -- so the newest built tag was used; verified is " +
    "true if the exact tag is built on quay.io, false if positively absent, null if it could not " +
    "be checked (non-biocontainer ref or a transient network error). " +
    "NEXT STEPS: if match_quality is exact_version and verified is not false, pass data.image as " +
    'the "container" when calling create_user_tool. If match_quality is name_only, the version you ' +
    "asked for wasn't among the built tags (or you didn't pin one) and the newest tag was " +
    "substituted -- show the user which image you got before using it. If image is null, or " +
    "verified is false, don't guess a tag: tell the user no built image was found for those " +
    "packages and ask how to proceed.",
  input,
  run,
  project: (out) => ({
    message: out.image
      ? `Resolved ${out.image} (${out.match_quality})`
      : "No biocontainer found for the requested packages",
  }),
};

register(recommendBiocontainerOp as AnyOperation);

export const recommendBiocontainer = (i: In, ctx: GalaxyContext) =>
  runOperation(recommendBiocontainerOp, i, ctx);
