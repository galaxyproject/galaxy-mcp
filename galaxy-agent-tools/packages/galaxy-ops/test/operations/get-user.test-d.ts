/**
 * What the exported user type promises, checked by the compiler rather than at runtime.
 *
 * `/api/users/current` answers `DetailedUserModel | AnonUserModel`, and the anonymous half
 * carries three disk-usage fields and nothing else -- no id, no email, no username. The tool
 * hands that reply back as it arrived, the way the other server does, so a type promising the
 * three as strings lets `(await getUser(...)).username.toLowerCase()` compile and then crash
 * on a session that never logged in. Only the compiler can hold this one: every runtime check
 * passes either way, which is why the type could drift from the value in the first place.
 */
import { describe, it, expectTypeOf } from "vitest";
import type { CurrentUser } from "../../src/operations/get-user";

describe("CurrentUser", () => {
  it("promises none of the three fields the anonymous reply leaves out", () => {
    expectTypeOf<CurrentUser["id"]>().toBeNullable();
    expectTypeOf<CurrentUser["email"]>().toBeNullable();
    expectTypeOf<CurrentUser["username"]>().toBeNullable();
  });

  it("still carries whatever else Galaxy sent", () => {
    // The record goes out whole, so a caller reading the quota or the disk usage does not
    // have to ask Galaxy a second time.
    expectTypeOf<CurrentUser["total_disk_usage"]>().toEqualTypeOf<unknown>();
  });
});
