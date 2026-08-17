#!/bin/sh
# Load a KEY=VALUE arguments file into the environment.
#
# Source it, do not execute it:
#     . "$PROJECT_DIR/scripts/utils/load_arguments.sh" "$ARGUMENTS_FILE"
#
# Reads the same file as src/util/arguments.py, so the shell scripts and the
# Python entry points cannot drift apart. ${NAME} placeholders are expanded
# against previously loaded keys and the existing environment, which is how
# ${FOLD_IDX} gets filled in from the value SLURM exports.
#
# Two caveats worth knowing:
#   - Values are expanded with eval, so this file is trusted input. Do not point
#     it at an arguments file you did not write.
#   - A '#' anywhere in a line starts a comment, so a value containing a literal
#     '#' must be quoted.

# Falls back to $ARGUMENTS_FILE because POSIX `.` does not portably pass
# arguments to a sourced script -- bash accepts `. file.sh arg`, dash ignores
# the arg and leaves $1 as the caller's. The env var works everywhere.
_load_arguments_file="${1:-${ARGUMENTS_FILE:-}}"

if [ -z "$_load_arguments_file" ]; then
    echo "load_arguments.sh: no arguments file given" >&2
    echo "  pass it as an argument or set ARGUMENTS_FILE" >&2
    return 1 2>/dev/null || exit 1
fi

if [ ! -f "$_load_arguments_file" ]; then
    echo "load_arguments.sh: arguments file not found: $_load_arguments_file" >&2
    return 1 2>/dev/null || exit 1
fi

_load_arguments_count=0

while IFS= read -r _line || [ -n "$_line" ]; do
    # Strip comments and surrounding whitespace
    _line=$(printf '%s' "$_line" | sed 's/#.*$//; s/^[[:space:]]*//; s/[[:space:]]*$//')

    [ -z "$_line" ] && continue

    case "$_line" in
        *=*) ;;
        *)
            echo "load_arguments.sh: skipping malformed line: $_line" >&2
            continue
            ;;
    esac

    _key=$(printf '%s' "${_line%%=*}" | sed 's/[[:space:]]*$//')
    _value=$(printf '%s' "${_line#*=}" | sed 's/^[[:space:]]*//')

    # Keys must be plain identifiers so they are safe to export
    case "$_key" in
        ''|*[!A-Za-z0-9_]*)
            echo "load_arguments.sh: skipping invalid key: $_key" >&2
            continue
            ;;
    esac

    # Strip one layer of surrounding quotes, matching the Python loader
    case "$_value" in
        \"*\") _value=$(printf '%s' "$_value" | sed 's/^"//; s/"$//') ;;
        \'*\') _value=$(printf '%s' "$_value" | sed "s/^'//; s/'$//") ;;
    esac

    # eval expands ${NAME} against what is already exported
    eval "export $_key=\"$_value\""
    _load_arguments_count=$((_load_arguments_count + 1))
done < "$_load_arguments_file"

echo "load_arguments.sh: loaded $_load_arguments_count entries from $_load_arguments_file"

# Make the path available to the Python entry points too, so they read the same
# file without every script having to pass --arguments-file explicitly.
export AUTO_QC_ARGUMENTS_FILE="$_load_arguments_file"

unset _line _key _value _load_arguments_file _load_arguments_count
