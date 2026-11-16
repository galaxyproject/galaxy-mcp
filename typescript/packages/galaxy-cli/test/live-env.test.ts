import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import setup from "./support/global-setup.js";

describe("the developer's live Galaxy credentials", () => {
  it("are not in the environment of an ordinary test file", () => {
    expect(Object.keys(process.env).filter((k) => k.startsWith("GXY_LIVE"))).toEqual([]);
    expect(process.env.GALAXY_API_KEY).toBeUndefined();
  });

  describe("global setup", () => {
    let teardown: (() => void) | undefined;
    // Setup sets GXY_TEST_ROOT; stubbing it first lets unstubAllEnvs put this file's value back.
    beforeEach(() => vi.stubEnv("GXY_TEST_ROOT", process.env.GXY_TEST_ROOT));
    afterEach(() => {
      teardown?.();
      vi.unstubAllEnvs();
    });

    it("provides them to the smoke test without putting them in process.env", () => {
      vi.stubEnv("GALAXY_URL", "http://127.0.0.1:9");
      vi.stubEnv("GALAXY_API_KEY", "canary-key");
      const provide = vi.fn();
      teardown = setup({ provide });
      expect(provide).toHaveBeenCalledWith("gxyLive", { url: "http://127.0.0.1:9", apiKey: "canary-key" });
      for (const [name, value] of Object.entries(process.env)) {
        if (name === "GALAXY_API_KEY") continue; // our own stub; safe-env.ts clears it in each file
        expect(value, name).not.toContain("canary-key");
      }
      expect(process.env.GXY_LIVE_API_KEY).toBeUndefined();
      expect(process.env.GXY_LIVE_URL).toBeUndefined();
    });

    it("accepts GXY_LIVE_* under those names and removes them from process.env", () => {
      vi.stubEnv("GXY_LIVE_URL", "http://127.0.0.1:9");
      vi.stubEnv("GXY_LIVE_API_KEY", "canary-key");
      const provide = vi.fn();
      teardown = setup({ provide });
      expect(provide).toHaveBeenCalledWith("gxyLive", { url: "http://127.0.0.1:9", apiKey: "canary-key" });
      expect(process.env.GXY_LIVE_API_KEY).toBeUndefined();
      expect(process.env.GXY_LIVE_URL).toBeUndefined();
    });

    it("provides nothing when they are not set", () => {
      vi.stubEnv("GALAXY_URL", "");
      vi.stubEnv("GALAXY_API_KEY", "");
      const provide = vi.fn();
      teardown = setup({ provide });
      expect(provide).toHaveBeenCalledWith("gxyLive", { url: undefined, apiKey: undefined });
    });
  });
});
