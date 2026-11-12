import { describe, it, expectTypeOf } from "vitest";
import type { UserToolRecord } from "../src/operations/get-user-tool";

describe("UserToolRecord", () => {
  it("accepts a Galaxy 26.0 record, which has no representation_errors or representation_status", () => {
    expectTypeOf<UserToolRecord["representation_errors"]>().toEqualTypeOf<string[] | undefined>();
    expectTypeOf<UserToolRecord["representation_status"]>().toEqualTypeOf<
      "ok" | "lifted" | "invalid" | undefined
    >();
    expectTypeOf<UserToolRecord["uuid"]>().toEqualTypeOf<string>();
  });
});
