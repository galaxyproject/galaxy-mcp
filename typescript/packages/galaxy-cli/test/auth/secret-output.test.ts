/**
 * Where a token may appear. It goes to stdout only from the commands whose job is to hand it
 * out, and nowhere else: not stderr, not an error, not a failed or half-finished run.
 */
import { describe, expect, it } from "vitest";
import { loginCommand } from "../../src/auth/commands/login";
import { logoutCommand } from "../../src/auth/commands/logout";
import { tokenCommand } from "../../src/auth/commands/token";
import { AuthError } from "../../src/auth/types";
import { FAKE_TOKEN, HOST, fakeLogin, harness } from "./harness";

const HOST_ARG = "https://dev.galaxy.example.org";
/** The harness's fakes for one run: `cap.out` and `cap.err` hold everything written to each stream. */
type H = ReturnType<typeof harness>;
/** One command invocation against a harness, resolving to its exit code. */
type Run = (h: H) => Promise<number>;

// `FAKE_TOKEN` is a canary: a string no other output could contain by accident, so finding it in a
// stream means the token went there. Each command is run once and both streams are searched.
describe("a stored, valid login", () => {
  // `handsOutToken` is written by hand for each command and checked both ways below: a command that
  // should print the token and does not fails as surely as one that prints it and should not.
  const cases: [name: string, handsOutToken: boolean, run: Run][] = [
    ["token", true, (h) => tokenCommand({}, h.deps)],
    ["token --json", true, (h) => tokenCommand({ json: true }, h.deps)],
    ["logout", false, (h) => logoutCommand({}, h.deps)],
  ];

  it.each(cases)("%s", async (_name, handsOutToken, run) => {
    const h = harness({ stored: [fakeLogin()] });
    await run(h);
    expect(h.cap.err).not.toContain(FAKE_TOKEN);
    expect(h.cap.out.includes(FAKE_TOKEN)).toBe(handsOutToken);
  });
});

// The error paths are where a message is most likely to quote the token, so every command is also
// run without one (expired, or none stored), and neither stream may carry it.
describe("when there is no usable login", () => {
  // The login expires before the harness's fixed clock, so it is expired without waiting.
  const expiredLogin = () =>
    harness({ stored: [fakeLogin({ expiresAt: "2026-11-06T09:00:00.000Z" })] });
  // With nothing stored there is no host to resolve, so the command is given one; it then reaches
  // its "no login" error and not the host-selection one.
  const noLogin = () => {
    const h = harness();
    h.deps.resolveHost = async () => HOST;
    return h;
  };
  const commands: [string, Run][] = [
    ["token", (h) => tokenCommand({}, h.deps)],
    ["token --json", (h) => tokenCommand({ json: true }, h.deps)],
    ["logout", (h) => logoutCommand({}, h.deps)],
  ];
  const situations: [string, () => H][] = [
    ["an expired login", expiredLogin],
    ["no login", noLogin],
  ];

  const matrix = commands.flatMap(([name, run]) =>
    situations.map(([what, make]) => [name, what, make, run] as const),
  );

  it.each(matrix)("%s with %s prints no token", async (_name, _what, make, run) => {
    const h = make();
    await run(h);
    expect(h.cap.err).not.toContain(FAKE_TOKEN);
    expect(h.cap.out).not.toContain(FAKE_TOKEN);
  });
});

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
