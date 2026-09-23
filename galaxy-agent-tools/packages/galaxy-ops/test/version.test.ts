import { describe, it, expect } from "vitest";
import {
  parseGalaxyVersion,
  parseRequirement,
  requirementSentence,
  satisfiesRequirement,
} from "../src/version";

const at = (raw: string) => {
  const v = parseGalaxyVersion(raw);
  if (!v) throw new Error(`${raw} did not parse`);
  return v;
};

describe("parseGalaxyVersion", () => {
  it("reads the two components Galaxy's version_major carries", () => {
    expect(parseGalaxyVersion("26.1")).toEqual({ raw: "26.1", major: 26, minor: 1 });
    expect(parseGalaxyVersion("26.0")).toEqual({ raw: "26.0", major: 26, minor: 0 });
  });

  it("takes the first two components of a fuller version string", () => {
    // A caller reading the release tag passes 26.1.1; a dev checkout answers 26.2.dev0.
    expect(parseGalaxyVersion("26.1.1")).toMatchObject({ major: 26, minor: 1 });
    expect(parseGalaxyVersion("26.2.dev0")).toMatchObject({ major: 26, minor: 2 });
    expect(parseGalaxyVersion("26.1-rc1")).toMatchObject({ major: 26, minor: 1 });
  });

  it("keeps what the server said, so a refusal can quote it", () => {
    expect(parseGalaxyVersion(" 26.1.1 ")?.raw).toBe("26.1.1");
  });

  it("is unknown rather than an error for anything it cannot read", () => {
    for (const bad of ["", "26", "twenty-six.one", "v26.1", ".1", undefined, null]) {
      expect(parseGalaxyVersion(bad)).toBeUndefined();
    }
  });
});

describe("parseRequirement", () => {
  it("accepts >=MAJOR.MINOR", () => {
    expect(parseRequirement(">=26.1")).toEqual({ major: 26, minor: 1 });
    expect(parseRequirement(">= 26.1")).toEqual({ major: 26, minor: 1 });
    expect(parseRequirement("  >=24.0  ")).toEqual({ major: 24, minor: 0 });
  });

  it("refuses anything else, including bounds version_major could never answer", () => {
    // A patch level lives in version_minor, which /api/version reports separately and which
    // a dev server fills with "dev0" -- a three-component bound could not be judged.
    for (const bad of [">=26.1.1", ">26.1", "26.1", "<=26.1", "^26.1", ">=26", ">=26.x", ""]) {
      expect(() => parseRequirement(bad)).toThrow(/">=MAJOR\.MINOR"/);
    }
  });
});

describe("satisfiesRequirement", () => {
  it("passes an equal or newer server", () => {
    expect(satisfiesRequirement(at("26.1"), ">=26.1")).toBe(true);
    expect(satisfiesRequirement(at("26.2"), ">=26.1")).toBe(true);
    expect(satisfiesRequirement(at("27.0"), ">=26.1")).toBe(true);
    expect(satisfiesRequirement(at("26.1.1"), ">=26.1")).toBe(true);
  });

  it("fails an older one, including a bigger minor under a smaller major", () => {
    expect(satisfiesRequirement(at("26.0"), ">=26.1")).toBe(false);
    expect(satisfiesRequirement(at("25.9"), ">=26.1")).toBe(false);
    expect(satisfiesRequirement(at("9.9"), ">=26.1")).toBe(false);
  });
});

describe("requirementSentence", () => {
  it("says the bound once, the same way for every surface", () => {
    expect(requirementSentence(">=26.1")).toBe("Requires Galaxy 26.1 or newer.");
    expect(requirementSentence(">= 24.0")).toBe("Requires Galaxy 24.0 or newer.");
  });
});
