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

# dash and some other POSIX shells ignore arguments passed to a sourced script,
# so `. load_arguments.sh myfile` silently sees no $1 there. Fall back to the
# ARGUMENTS_FILE environment variable, which the SLURM scripts already export.
_load_arguments_file="${1:-${ARGUMENTS_FILE:-}}"

if [ -z "$_load_arguments_file" ]; then
    echo "load_arguments.sh: no arguments file given" >&2
    return 1 2>/dev/null || exit 1
fi

if [ ! -f "$_load_arguments_file" ]; then
    echo "load_arguments.sh: arguments file not found: $_load_arguments_file" >&2
    return 1 2>/dev/null || exit 1
fi

_load_arguments_count=0
_load_arguments_warnings=0
_load_arguments_external=""

# Every key the file defines, gathered up front so a reference can be told
# apart from a forward reference to a key defined further down.
_load_arguments_keys=$(sed 's/#.*$//' "$_load_arguments_file" \
    | grep -o '^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=' \
    | sed 's/[[:space:]]//g; s/=$//')

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

    # An undefined ${NAME} expands to nothing rather than failing, so a forward
    # reference or a typo silently yields a truncated value: with SCRATCH_DIR
    # defined further down the file, PROJECT_DIR=${SCRATCH_DIR}/automated-qc
    # becomes "/automated-qc" -- a plausible-looking absolute path whose first
    # symptom appears much later, somewhere unrelated. Checking the expansion
    # is not enough (that string is neither empty nor malformed), so inspect
    # each referenced name directly. The Python loader warns about this too.
    #
    # Precision matters here: FOLD_IDX is deliberately supplied by SLURM at
    # submit time and never defined in the file, so warning on every unset
    # reference would fire five times per load and teach everyone to ignore
    # the output. A reference is only certainly wrong when the name IS defined
    # in this file but further down -- that is the forward-reference bug.
    # Names defined nowhere are collected and mentioned once at the end, since
    # they are usually environment-supplied and occasionally a typo.
    case "$_value" in
        *'${'*)
            for _ref in $(printf '%s' "$_value" \
                          | grep -o '\${[A-Za-z_][A-Za-z0-9_]*}' \
                          | sed 's/[${}]//g'); do
                eval "_refval=\${$_ref:-}"
                [ -n "$_refval" ] && continue

                if printf '%s\n' "$_load_arguments_keys" | grep -qx "$_ref"; then
                    echo "load_arguments.sh: WARNING: $_key references" \
                         "\${$_ref}, which is defined LATER in this file and" \
                         "so expands to nothing here. Move $_ref above $_key." >&2
                    _load_arguments_warnings=$((_load_arguments_warnings + 1))
                else
                    _load_arguments_external="$_load_arguments_external $_ref"
                fi
            done
            unset _ref _refval
            ;;
    esac

    # eval expands ${NAME} against what is already exported
    eval "export $_key=\"$_value\""

    _load_arguments_count=$((_load_arguments_count + 1))
done < "$_load_arguments_file"

if [ "$_load_arguments_warnings" -gt 0 ]; then
    echo "load_arguments.sh: $_load_arguments_warnings forward reference(s)" \
         "above -- those values are truncated, not what you wrote" >&2
fi

if [ -n "$_load_arguments_external" ]; then
    # Expected for FOLD_IDX, which SLURM exports per job. Listed rather than
    # warned about, because interactively it is simply not set yet.
    echo "load_arguments.sh: expanded from the environment (empty if unset):" \
         "$(printf '%s\n' $_load_arguments_external | sort -u | tr '\n' ' ')" >&2
fi

echo "load_arguments.sh: loaded $_load_arguments_count entries from $_load_arguments_file"

# Make the path available to the Python entry points too, so they read the same
# file without every script having to pass --arguments-file explicitly.
export AUTO_QC_ARGUMENTS_FILE="$_load_arguments_file"

unset _line _key _value _load_arguments_file _load_arguments_count \
      _load_arguments_warnings _load_arguments_external _load_arguments_keys
