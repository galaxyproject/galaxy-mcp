/**
 * Where a token may appear: nowhere. Not stdout, not stderr, not an error, not a failed or
 * half-finished run.
 */
import { describe, expect, it } from "vitest";
import { loginCommand } from "../../src/auth/commands/login";
import { AuthError } from "../../src/auth/types";
import { FAKE_TOKEN, harness } from "./harness";

const HOST_ARG = "https://dev.galaxy.example.org";

// `FAKE_TOKEN` is a canary: a string no other output could contain by accident, so finding it in a
// stream means the token went there. `cap.out` and `cap.err` hold everything the harness's fakes
// were given to write to each stream.
//
// `login` never hands out a token, so its stdout must be empty as well as free of it. The three runs
// are its three outcomes: a key succeeding, a key Galaxy refuses, and an admin key that login refuses
// to put in a file without `--insecure-storage`.
describe("login", () => {
  it("never prints the token, whether it succeeds, is refused, or cannot store", async () => {
    const stdinText = `${FAKE_TOKEN}\n`;
    const accepted = harness({ stdinText });
    const refused = harness({ stdinText });
    refused.knobs.whoami = new AuthError("not_logged_in", "Galaxy rejected the key");
    const unstorable = harness({ stdinText });
    unstorable.knobs.whoami = { username: "root", isAdmin: true };

    await loginCommand({ host: HOST_ARG, withToken: true }, accepted.deps);
    await loginCommand({ host: HOST_ARG, withToken: true }, refused.deps);
    await loginCommand({ host: HOST_ARG, withToken: true }, unstorable.deps);

    for (const h of [accepted, refused, unstorable]) {
      expect(h.cap.out).toBe("");
      expect(h.cap.err).not.toContain(FAKE_TOKEN);
    }
  });
});
