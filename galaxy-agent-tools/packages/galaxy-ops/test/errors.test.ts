import { describe, it, expect } from "vitest";
import {
  GalaxyAuthError,
  GalaxyNotFoundError,
  GalaxyConnectionError,
  ToolRequestRejectedError,
  JobFailedError,
  classifyHttp,
} from "../src/errors";

describe("error classification", () => {
  it("maps 401/403 -> auth, 404 -> not found", () => {
    expect(classifyHttp(401, undefined)).toBeInstanceOf(GalaxyAuthError);
    expect(classifyHttp(403, undefined)).toBeInstanceOf(GalaxyAuthError);
    expect(classifyHttp(404, undefined)).toBeInstanceOf(GalaxyNotFoundError);
  });
  it("maps other non-2xx to a generic connection error carrying status", () => {
    const e = classifyHttp(500, { err_msg: "boom" });
    expect(e).toBeInstanceOf(GalaxyConnectionError);
    expect((e as GalaxyConnectionError).status).toBe(500);
  });
  it("ToolRequestRejectedError carries toolId + err_msg", () => {
    const e = new ToolRequestRejectedError("fastqc/0.74", "could not expand");
    expect(e.toolId).toBe("fastqc/0.74");
    expect(e.message).toContain("could not expand");
  });
  it("JobFailedError carries job id + terminal state + stderr", () => {
    const e = new JobFailedError("job1", "error", "traceback");
    expect(e.jobId).toBe("job1");
    expect(e.state).toBe("error");
    expect(e.stderr).toBe("traceback");
  });
});
