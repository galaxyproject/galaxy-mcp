# @galaxyproject/galaxy-cli

Galaxy agent operations on the command line. One subcommand per operation --
histories, datasets, tools, workflows, pages, and the IWC catalog -- with table / JSON
output and meaningful exit codes. Built on
[`@galaxyproject/galaxy-ops`](https://www.npmjs.com/package/@galaxyproject/galaxy-ops).

## Install

```bash
npm install -g @galaxyproject/galaxy-cli   # provides `galaxy-cli` and `gxy` (the same program, shorter name)
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

## Logging in

`gxy auth` keeps a login for a Galaxy and hands its token to scripts. `gxy` and `galaxy-cli` are the
same program. Other commands do not use a stored login yet: set `GALAXY_URL` and `GALAXY_API_KEY` as
before.

```bash
printf %s "$KEY" | gxy auth login --host https://usegalaxy.org --with-token   # store an API key
gxy auth token                                                                 # print the stored token
gxy auth logout                                                                # delete the stored login
```

| Command | Does |
| --- | --- |
| `auth login` | store an API key read from stdin (`--with-token`, required); `--insecure-storage` allows an admin key |
| `auth token` | print the token on stdout (`--json` for an object with its expiry) |
| `auth logout` | delete the stored login; a pasted API key is not revoked |

Create an API key in Galaxy's user preferences and pipe it in:
`printf %s "$KEY" | gxy auth login --host <url> --with-token`. At a terminal, `--with-token` asks
for the key without echoing it. If stdin is a terminal but stderr is redirected, where the prompt
would go unseen, it refuses (exit 64) and asks you to pipe the key in. `gxy auth login` without
`--with-token` says the same and exits 64. `auth logout` only forgets a key, because it may be in use
elsewhere: it keeps working until you revoke it in Galaxy's user preferences, and `auth logout` says
so.

### Where logins are kept

Logins live in `hosts.yml` in the config directory (`$GXY_CONFIG_DIR`, else
`$XDG_CONFIG_HOME/gxy`, else `~/.config/gxy`; a relative path in either variable is ignored), token
and all, in plain text, readable only by you. A key that carries admin rights is the whole server,
so `gxy auth login` refuses it and stores nothing (exit 77), unless you add `--insecure-storage` to
put it in `hosts.yml`.

`gxy` refuses a `hosts.yml` that other users can write to (it would not know what they put in it),
that is a symlink or belongs to someone else, and one it cannot make sense of rather than overwrite
it. A file or config directory that others can read or enter is narrowed to 600 or 700 with a
warning. An entry written by a different version of `gxy` (a storage or method this one does not
know) is skipped with a warning and kept in the file as it is, so it does not stop your other
logins from working; asking for that very Galaxy fails and says it was written by a newer version,
rather than that you are not logged in. Each refusal exits 65 (not 77, which would suggest logging in
again, and that meets the same refusal), names the file and what to do, and changes nothing.

### Which Galaxy the `auth` commands use

`auth token` and `auth logout` pick their Galaxy by `--host` (the same as `--url`), then the
`--profile` URL, then `GALAXY_URL` in the environment, then `GALAXY_URL` in `.env`, then the only
stored login. They never use a key from `--api-key`, `GALAXY_API_KEY`, `.env` or a profile, and warn
once on stderr if `--api-key` is given (and, for `auth login` only, `--profile`). `auth login` never
takes its host from `.env` or a profile (a repository must not choose where your key goes): give
`--host` or set `GALAXY_URL`.

### What is checked before a key is sent

Before a key goes anywhere, `gxy` checks that the address answers as a Galaxy (`/api/version`),
and it never follows a redirect with a key attached: a redirect is reported as a wrong address,
with the place it points to. Plain `http://` is refused (exit 76) for any host but this machine
(`localhost`, `127.0.0.0/8`, `::1`), because a token would cross the network in the clear.

`auth login` checks a Galaxy with two requests (a version probe, then the user lookup), and each
gives up after 15 seconds. Ctrl-C stops a check at once.

| Variable | Does |
| --- | --- |
| `GXY_CONFIG_DIR` | Where `hosts.yml` lives (see above). A relative path is ignored, so the file does not follow the working directory |

Behind a proxy, set `NODE_USE_ENV_PROXY=1` along with `HTTPS_PROXY` (and `NO_PROXY` for a Galaxy on
your own network). Node's `fetch` ignores the proxy variables without it, and reads it only at
startup, so it has to be in the environment before `gxy` runs. It needs Node 22.21+ or 24+. If
`gxy` cannot reach the Galaxy while a proxy variable is set and the flag is not, it says so (not when
the Galaxy answered with an error, which a proxy being ignored would not cause). For a Galaxy with a
certificate from a private CA, set `NODE_EXTRA_CA_CERTS=/path/to/ca.pem`. Login never offers to
switch off certificate checks.

```bash
printf %s "$KEY" | NODE_USE_ENV_PROXY=1 HTTPS_PROXY=http://proxy.example.org:3128 NO_PROXY=localhost gxy auth login --host https://galaxy.example.org --with-token
```

## Documentation

See the [TypeScript workspace README](https://github.com/galaxyproject/galaxy-mcp/tree/main/typescript#readme)
for the full operation list, output formats, and exit-code reference.

## License

[MIT](https://github.com/galaxyproject/galaxy-mcp/blob/main/LICENSE)
