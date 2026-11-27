// typescript-eslint cannot use TypeScript 7 yet (typescript-eslint/typescript-eslint#10940). Its
// experimental `tsgo` backend (see galaxy-cli's eslint.config.js) wants TypeScript 6 as
// `typescript` and TypeScript 7.1 as `@typescript/native`. The workspace's own TypeScript is 7,
// so pnpm would hand typescript-eslint that one. This gives typescript-eslint's packages their own
// copies instead. Only the linter sees them: tsc, tsup and the tests keep resolving TypeScript 7.
//
// The versions here are ranges on purpose, so the linter follows TypeScript 6 and 7.1 as they
// release (the lockfile moves on `pnpm update`). Drop this file once typescript-eslint supports
// TypeScript 7 normally.
const tseslintPackages = /^(typescript-eslint|@typescript-eslint\/.+)$/;

function readPackage(pkg) {
  if (!tseslintPackages.test(pkg.name) || !pkg.peerDependencies?.typescript) return pkg;

  delete pkg.peerDependencies.typescript;
  pkg.dependencies = { ...pkg.dependencies, typescript: "^6" };

  if (pkg.peerDependencies["@typescript/native"]) {
    delete pkg.peerDependencies["@typescript/native"];
    pkg.dependencies["@typescript/native"] = "npm:typescript@^7.1.0-beta";
  }
  return pkg;
}

module.exports = { hooks: { readPackage } };
