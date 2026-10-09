import { describe, it, expect } from "vitest";
import { createGalaxyContext, invokeWorkflowOp, runWithEnvelope } from "@galaxyproject/galaxy-ops";
import { exitCodeFor, EX_USAGE, EX_SOFTWARE, EX_PROTOCOL, EX_UNAVAILABLE } from "../src/exit";

describe("exitCodeFor", () => {
  it("maps error kinds to BSD sysexits", () => {
    expect(exitCodeFor(undefined)).toBe(0);
    expect(exitCodeFor("auth")).toBe(77);
    expect(exitCodeFor("not_found")).toBe(66);
    expect(exitCodeFor("connection")).toBe(69);
    expect(exitCodeFor("version")).toBe(76);
    expect(exitCodeFor("tool_request_rejected")).toBe(65);
    expect(exitCodeFor("job_failed")).toBe(70);
    expect(exitCodeFor("unknown")).toBe(70);
  });
  it("exposes usage + software constants", () => {
    expect(EX_USAGE).toBe(64);
    expect(EX_SOFTWARE).toBe(70);
    expect(EX_PROTOCOL).toBe(76);
  });
  it("keeps a too-old server apart from an unreachable one", () => {
    // Both are the server's fault, but only one is worth retrying.
    expect(exitCodeFor("version")).not.toBe(exitCodeFor("connection"));
  });
});

describe("a rejected argument is a usage error, not an outage", () => {
  it("maps validation to EX_USAGE rather than EX_UNAVAILABLE", () => {
    expect(exitCodeFor("validation")).toBe(EX_USAGE);
    expect(exitCodeFor("validation")).not.toBe(EX_UNAVAILABLE);
  });

  it("carries an op's own rejected argument all the way to exit 64", async () => {
    // The whole chain, not the map on its own: invoke_workflow refuses the argument, the
    // envelope carries the kind out, and the CLI turns that kind into a status. Malformed
    // inputs used to arrive here as "connection" and exit 69, which tells an agent to retry
    // something that can never succeed.
    const ctx = createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      fetchImpl: (async () => {
        throw new Error("no request should be made for an argument the op already refused");
      }) as typeof fetch,
    });
    const result = await runWithEnvelope(
      invokeWorkflowOp as never,
      { workflowId: "wf1", inputs: "{not json" } as never,
      ctx,
    );
    expect(result.success).toBe(false);
    expect(result.message).toMatch(/^inputs must be a JSON object string or an object/);
    expect(result.errorKind).toBe("validation");
    expect(exitCodeFor(result.errorKind)).toBe(EX_USAGE);
    expect(exitCodeFor(result.errorKind)).not.toBe(EX_UNAVAILABLE);
  });
});
