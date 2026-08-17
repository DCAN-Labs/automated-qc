"""Load run configuration from a plain KEY=VALUE arguments file.

Motivation: cluster paths were hardcoded across the SLURM scripts and duplicated
between training and inference, so moving the project or running as a different
user meant editing a dozen files. A single arguments file is read by both the
shell scripts (via scripts/utils/load_arguments.sh) and the Python entry points
(via this module), so the two can never disagree.

File format:

    # comments start with a hash
    PROJECT_DIR=/users/1/<x500>/projects/automated-qc
    SCRATCH_DIR=/scratch.global/<x500>/auto_qc
    MODEL_NAME=model_04d0

    # ${...} interpolates earlier keys, then the environment
    FOLDER=${SCRATCH_DIR}/fmaps/
    MODEL_SAVE_LOCATION=${SCRATCH_DIR}/${MODEL_NAME}/${MODEL_NAME}_fold_${FOLD_IDX}.pt

Keys map to argparse destinations by lowercasing: FOLDER sets --folder,
CSV_INPUT_FILE sets --csv-input-file, and so on. Keys with no matching argument
(PROJECT_DIR, SLURM_ACCOUNT, ...) are ignored here and consumed by the shell
loader instead.

Precedence is: explicit command line > arguments file > argparse default. That
ordering matters -- it means the arguments file sets the baseline for a study
while a one-off run can still override a single value on the command line
without editing shared config.
"""

import argparse
import logging
import os
import re

log = logging.getLogger(__name__)

# Matches ${NAME} placeholders for interpolation.
PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Keys must be a plain identifier so they can round-trip through the shell.
KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off", ""}


class ArgumentsFileError(Exception):
    """Raised when an arguments file cannot be parsed or resolved."""


def parse_bool(text):
    """Interpret a config string as a boolean, for store_true style flags."""
    normalized = str(text).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ArgumentsFileError(
        f"Cannot interpret {text!r} as a boolean. Use true/false, yes/no, or 1/0."
    )


def interpolate(value, resolved, source_key):
    """Expand ${NAME} against already-resolved keys, then the environment.

    Unresolved placeholders are left literal rather than substituted with an
    empty string, and a warning is emitted. Silently producing
    'model_fold_.pt' from an unset ${FOLD_IDX} is far harder to notice than a
    filename with a visible ${FOLD_IDX} in it.
    """
    missing = []

    def _replace(match):
        name = match.group(1)
        if name in resolved:
            return resolved[name]
        if name in os.environ:
            return os.environ[name]
        missing.append(name)
        return match.group(0)

    result = PLACEHOLDER_RE.sub(_replace, value)

    if missing:
        log.warning(
            f"{source_key}: unresolved placeholder(s) {', '.join(sorted(set(missing)))} "
            f"-- left literal in {result!r}. Set them in the arguments file or "
            "export them before the run."
        )

    return result


def load_arguments(path):
    """Parse an arguments file into an ordered dict of resolved strings."""
    if not os.path.exists(path):
        raise ArgumentsFileError(f"Arguments file not found: {path}")

    resolved = {}

    with open(path) as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue

            if "=" not in line:
                raise ArgumentsFileError(
                    f"{path}:{lineno}: expected KEY=VALUE, got {raw_line.strip()!r}"
                )

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Allow quoting for values with trailing spaces or hashes.
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]

            if not KEY_RE.match(key):
                raise ArgumentsFileError(
                    f"{path}:{lineno}: invalid key {key!r}. Keys must be plain "
                    "identifiers so the shell loader can export them."
                )

            resolved[key] = interpolate(value, resolved, f"{path}:{lineno} ({key})")

    log.info(f"Loaded {len(resolved)} entries from {path}")
    return resolved


def coerce_for_action(action, raw_value, key):
    """Convert a config string to the type the argparse action expects."""
    if isinstance(action, argparse._StoreTrueAction):
        return parse_bool(raw_value)
    if isinstance(action, argparse._StoreFalseAction):
        return not parse_bool(raw_value)

    if raw_value == "" and action.type is not None:
        # An empty value means "leave the argparse default alone".
        return None

    if action.type is not None:
        try:
            return action.type(raw_value)
        except (TypeError, ValueError) as exc:
            raise ArgumentsFileError(
                f"{key}={raw_value!r} is not valid for --{action.dest.replace('_', '-')}: {exc}"
            )

    return raw_value


def apply_to_parser(parser, mapping):
    """Set argparse defaults from the mapping. Returns the keys actually applied.

    Anything without a matching argument destination is ignored -- those keys
    exist for the shell scripts (PROJECT_DIR, SLURM_ACCOUNT, VENV_PYTHON, ...).
    """
    # Index by both the exact dest and its lowercase form. Most dests are already
    # lowercase, but --DEBUG has an uppercase dest, and a key that silently fails
    # to apply is worse than one that errors.
    actions_by_dest = {}
    for action in parser._actions:
        actions_by_dest.setdefault(action.dest, action)
        actions_by_dest.setdefault(action.dest.lower(), action)

    applied, ignored = {}, []

    for key, raw_value in mapping.items():
        dest = key.lower()
        action = actions_by_dest.get(dest) or actions_by_dest.get(key)

        if action is None:
            ignored.append(key)
            continue

        value = coerce_for_action(action, raw_value, key)
        if value is None and not isinstance(
            action, (argparse._StoreTrueAction, argparse._StoreFalseAction)
        ):
            continue

        # set_defaults bypasses argparse's own choices validation, so check here
        # rather than letting a typo surface as a confusing failure much later.
        if action.choices is not None and value not in action.choices:
            raise ArgumentsFileError(
                f"{key}={raw_value!r} is not one of {sorted(action.choices)}"
            )

        # Store under the action's real dest, not the lowercased key
        applied[action.dest] = value

    parser.set_defaults(**applied)

    if applied:
        log.info(
            "Applied from arguments file: "
            + ", ".join(sorted(applied.keys()))
        )
    if ignored:
        log.debug(
            "Keys with no matching argument (used by the shell scripts): "
            + ", ".join(sorted(ignored))
        )

    return applied


def add_arguments_file_option(parser):
    """Register the --arguments-file option on a parser."""
    parser.add_argument(
        "--arguments-file",
        default=None,
        help="Path to a KEY=VALUE arguments file supplying defaults for the "
        "options below. Anything given on the command line overrides it. "
        "Falls back to the AUTO_QC_ARGUMENTS_FILE environment variable.",
    )


def resolve_arguments_file(parser, argv):
    """Find the arguments file from argv or the environment, and apply it.

    Runs a permissive pre-parse so the file can be located before the real parse
    happens, which is what gives command line arguments precedence over it.
    """
    known, _ = parser.parse_known_args(argv)

    path = getattr(known, "arguments_file", None) or os.environ.get(
        "AUTO_QC_ARGUMENTS_FILE"
    )

    if not path:
        return None, {}

    mapping = load_arguments(path)
    apply_to_parser(parser, mapping)

    return path, mapping
