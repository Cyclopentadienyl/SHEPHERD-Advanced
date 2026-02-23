#!/usr/bin/env bash
###############################################################################
# SHEPHERD-Advanced Launcher Entry Point (Linux/ARM)
###############################################################################
# 
# This is a convenience wrapper that:
# 1. Activates the virtual environment
# 2. Passes all arguments to scripts/launch/shep_launch.py
# 3. Allows users to set COMMANDLINE_ARGS environment variable
# 
# Usage Examples:
#   ./launch_shepherd.sh
#   ./launch_shepherd.sh --xformers
#   ./launch_shepherd.sh --print-plan
#   ./launch_shepherd.sh --dry-run
# 
#   export COMMANDLINE_ARGS="--xformers --cudnn-sdpa"
#   ./launch_shepherd.sh
###############################################################################

set -euo pipefail

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

log() {
    echo -e "${CYAN}[SHEPHERD]${NC} $*"
}

success() {
    echo -e "${GREEN}  ✓${NC} $*"
}

error() {
    echo -e "${RED}ERROR:${NC} $*"
    exit 1
}

warning() {
    echo -e "${YELLOW}WARNING:${NC} $*"
}

###############################################################################
# Main
###############################################################################
echo -e "${CYAN}SHEPHERD-Advanced Launcher${NC}"
echo ""

# Check if virtual environment exists
if [[ ! -f ".venv/bin/activate" ]]; then
    error "Virtual environment not found! Please run deploy.sh first"
fi

# Activate virtual environment
log "Activating virtual environment..."
source .venv/bin/activate
success "Virtual environment activated"
echo ""

# Check if shep_launch.py exists
if [[ ! -f "scripts/launch/shep_launch.py" ]]; then
    error "scripts/launch/shep_launch.py not found!"
    warning "Expected file location: scripts/launch/shep_launch.py"
fi

# Launch the Python launcher with all arguments
log "Launching SHEPHERD via shep_launch.py..."
echo ""
python scripts/launch/shep_launch.py "$@"

# Capture exit code
EXIT_CODE=$?

# Deactivate virtual environment
deactivate

exit $EXIT_CODE
