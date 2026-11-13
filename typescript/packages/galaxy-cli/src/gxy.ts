import { buildProgram } from "./program";

// The same program under its short name. It is a second entry file, not a check of `argv[1]`:
// pnpm and the Windows shims start `node dist/gxy.js`, so the file is the only thing that says
// which name the person typed.
await buildProgram({ invokedAs: "gxy" }).parseAsync(process.argv);
