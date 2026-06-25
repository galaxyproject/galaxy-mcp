# @galaxyproject/galaxy-cli

Galaxy agent operations on the command line. One subcommand per operation --
histories, datasets, tools, workflows, and the IWC catalog -- with table / JSON
output and meaningful exit codes. Built on
[`@galaxyproject/galaxy-ops`](https://www.npmjs.com/package/@galaxyproject/galaxy-ops).

## Install

```bash
npm install -g @galaxyproject/galaxy-cli   # provides the `galaxy-cli` command
# or run without installing:
npx @galaxyproject/galaxy-cli --help
```

Requires Node.js `>=22.19`.

## Usage

Set your [Galaxy](https://galaxyproject.org/) connection, then run a command:

```bash
export GALAXY_URL=https://usegalaxy.org/
export GALAXY_API_KEY=your-api-key

galaxy-cli get_user
galaxy-cli --format json get_histories --name rnaseq
galaxy-cli get_workflow_input_template <workflowId>
galaxy-cli invoke_workflow <workflowId> --inputs @inputs.json --history-name "WF run"
```

Credentials resolve from (first match wins): `--url`/`--api-key` flags, the
`GALAXY_URL`/`GALAXY_API_KEY` environment variables, a `.env` file in the current
directory, or a planemo profile (`--profile <name>`). Output is a table by
default; `--format json` prints the full result envelope for scripting. Exit
codes follow `sysexits.h` conventions (e.g. `77` for auth failures).

Run `galaxy-cli --help`, or `galaxy-cli <command> --help`, for the full list of
commands and their arguments.

## Documentation

See the [galaxy-agent-tools workspace README](https://github.com/galaxyproject/galaxy-mcp/tree/main/galaxy-agent-tools#readme)
for the full operation list, output formats, and exit-code reference.

## License

[MIT](https://github.com/galaxyproject/galaxy-mcp/blob/main/LICENSE)
