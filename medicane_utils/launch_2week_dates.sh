#!/bin/bash

SCRIPT_NAME="download_airmassRGB.py"
DATE_FORMAT="%Y-%m-%d %H:%M"
START_DATE="2015-01-01 00:00"
END_BOUND="2015-12-31 23:59"
INTERVAL_DAYS=14

run_script_for_range() {
    local start_date="$1"
    local end_date="$2"

    if ! python "$SCRIPT_NAME" --start "$start_date" --end "$end_date"; then
        printf "Errore: esecuzione fallita per l'intervallo %s -> %s\n" "$start_date" "$end_date" >&2
        return 1
    fi
}

date_to_timestamp() {
    local date_input="$1"
    local timestamp

    if ! timestamp=$(date -d "$date_input" +%s 2>/dev/null); then
        printf "Errore: formato data non valido '%s'\n" "$date_input" >&2
        return 1
    fi

    printf "%s" "$timestamp"
}

timestamp_to_date() {
    local ts="$1"
    local date_str

    if ! date_str=$(date -d "@$ts" +"$DATE_FORMAT" 2>/dev/null); then
        printf "Errore: timestamp non valido '%s'\n" "$ts" >&2
        return 1
    fi

    printf "%s" "$date_str"
}

main() {
    local current_start_ts; current_start_ts=$(date_to_timestamp "$START_DATE") || return 1
    local end_bound_ts; end_bound_ts=$(date_to_timestamp "$END_BOUND") || return 1
    local interval_sec=$((INTERVAL_DAYS * 86400))

    while [[ "$current_start_ts" -lt "$end_bound_ts" ]]; do
        local current_end_ts=$((current_start_ts + interval_sec - 60))

        if [[ "$current_end_ts" -gt "$end_bound_ts" ]]; then
            current_end_ts=$end_bound_ts
        fi

        local start_str; start_str=$(timestamp_to_date "$current_start_ts") || return 1
        local end_str; end_str=$(timestamp_to_date "$current_end_ts") || return 1

        run_script_for_range "$start_str" "$end_str" || return 1

        current_start_ts=$((current_end_ts + 60))
    done
}

main

