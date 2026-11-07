#!/bin/bash
# See FLOW_MATCHING.md for the common recipe and overrides.
exec bash "$(dirname "${BASH_SOURCE[0]}")/_flow_common.sh" image_noise eval "$@"
