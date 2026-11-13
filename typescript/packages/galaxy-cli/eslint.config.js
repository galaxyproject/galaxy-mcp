// Lint and format cover the `gxy auth` code only: src/auth, test/auth and test/support. The
// other files here are not listed in any `files` glob, so ESLint never looks at them.
import tseslint from "typescript-eslint";
import regexp from "eslint-plugin-regexp";
import nodePlugin from "eslint-plugin-n";
import vitest from "@vitest/eslint-plugin";

const authSource = ["src/auth/**/*.ts"];
const authTests = ["test/auth/**/*.ts", "test/support/**/*.ts"];
const authFiles = [...authSource, ...authTests];

// Files that legitimately read the real process environment: the wiring that builds `deps.env`,
// and the helpers whose default parameter is the real environment. Commands take `deps.env`.
const mayReadProcessEnv = [
  "src/auth/deps.ts",
  "src/auth/paths.ts",
  "src/auth/browser.ts",
  "src/auth/store.ts",
  "src/auth/keyring-store.ts",
];

export default tseslint.config(
  { ignores: ["dist/**", "node_modules/**"] },
  {
    linterOptions: { reportUnusedDisableDirectives: "error" },
  },

  // Type-aware strict rules.
  {
    files: authFiles,
    extends: [
      tseslint.configs.strictTypeChecked,
      regexp.configs["flat/recommended"],
      nodePlugin.configs["flat/recommended-module"],
    ],
    languageOptions: {
      parserOptions: {
        // test/tsconfig.json extends tsconfig.test.json, so the tests have a project too.
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      "@typescript-eslint/switch-exhaustiveness-check": "error",
      "@typescript-eslint/consistent-type-imports": "error",

      // Numbers in a template are fine: `${n} ms` cannot print "[object Object]" or "undefined",
      // which is what this rule is for. Everything else it bans stays banned.
      "@typescript-eslint/restrict-template-expressions": ["error", { allowNumber: true }],
      // `const { token: _token, ...rest } = x` is how a field is left out of a copy.
      "@typescript-eslint/no-unused-vars": ["error", { ignoreRestSiblings: true }],

      // Security-relevant core rules.
      "no-eval": "error",
      "no-implied-eval": "error",
      "no-new-func": "error",
      eqeqeq: ["error", "always"],

      // TypeScript resolves imports (extensionless, the workspace alias), and the package is
      // bundled by tsup, so these two Node-resolution rules only produce noise.
      "n/no-missing-import": "off",
      "n/no-unpublished-import": "off",
    },
  },

  // Commands write through `deps.io` and read `deps.env`.
  {
    files: authSource,
    rules: {
      "no-console": "error",
      "no-restricted-properties": [
        "error",
        {
          object: "process",
          property: "env",
          message: "Read the injected `deps.env`; only the wiring files may touch process.env.",
        },
      ],
    },
  },
  { files: mayReadProcessEnv, rules: { "no-restricted-properties": "off" } },

  // Tests.
  {
    files: authTests,
    extends: [vitest.configs.recommended],
    rules: {
      // Fakes that stand in for an async interface are `async` without awaiting anything, so
      // that a throw becomes a rejection as it would in the real thing; and `x!` on a lookup the
      // test has just arranged fails the test with a TypeError if it is wrong.
      "@typescript-eslint/require-await": "off",
      "@typescript-eslint/no-non-null-assertion": "off",
      "vitest/expect-expect": [
        "error",
        { assertFunctionNames: ["expect", "expectTokenOnlyWhereAllowed", "expectCancelled"] },
      ],
      "vitest/no-conditional-expect": "error",
      "vitest/no-focused-tests": "error",
      // vitest's `expect(value, message)` takes a second argument.
      "vitest/valid-expect": ["error", { maxArgs: 2 }],
      "vitest/no-disabled-tests": "warn",
    },
  },
);
