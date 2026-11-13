import { buildProgram } from "./program";

await buildProgram({ invokedAs: "galaxy-cli" }).parseAsync(process.argv);
