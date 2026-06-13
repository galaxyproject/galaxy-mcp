import { describe, it, expectTypeOf } from "vitest";
import type { GetJson } from "../src/bindings";

describe("bindings response extractors", () => {
  it("GetJson resolves a 200 application/json body for a known path", () => {
    // /api/version returns an object; the extractor must not be `never`.
    expectTypeOf<GetJson<"/api/version">>().not.toBeNever();
  });
});
