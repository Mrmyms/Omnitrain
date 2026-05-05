#!/bin/bash

# OmniTrain v2.1.0 - Industrial Setup Script
# "Fuse Everything. Trust Nothing. Verify Formally."

set -e

# Colors for terminal output
BLUE='\033[0;34m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "   .---."
echo "  ( @ @ )  OmniTrain Industrial Intelligence"
echo "   )   (   Setup & Industrialization"
echo "  /|||||\\"
echo "  \" \" \" \""
echo -e "${NC}"

echo -e "${BLUE}[1/4]${NC} Checking environment..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR:${NC} python3 not found. Please install Python 3.10+."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "  Found Python ${GREEN}${PYTHON_VERSION}${NC}"

echo -e "\n${BLUE}[2/4]${NC} Creating virtual environment (.venv)..."
if [ -d ".venv" ]; then
    echo -e "  ${YELLOW}Notice:${NC} .venv already exists. Skipping creation."
else
    python3 -m venv .venv
    echo -e "  ${GREEN}Virtual environment created.${NC}"
fi

echo -e "\n${BLUE}[3/4]${NC} Installing dependencies..."
./.venv/bin/python3 -m pip install --upgrade pip
./.venv/bin/python3 -m pip install -r requirements.txt
./.venv/bin/python3 -m pip install -e .

echo -e "\n${BLUE}[4/4]${NC} Industrialization Audit..."
if ./.venv/bin/python3 tests/test_integrity.py; then
    echo -e "\n  ${GREEN}Audit Passed: Integrity Verification Successful.${NC}"
else
    echo -e "\n  ${RED}Audit Failed: Please check the logs above.${NC}"
    exit 1
fi

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}  OMNITRAIN READY FOR INDUSTRIAL DEPLOYMENT${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "\nQuick Start Commands:"
echo -e "  ${CYAN}omni${NC}          - Launch the BioLiquid Dashboard"
echo -e "  ${CYAN}omni --help${NC}   - View CLI reference"

# Optional: Add alias to shell profile
echo -e "\n${YELLOW}PRO-TIP:${NC} Would you like to add the '1omni' alias to your shell profile? (y/n)"
read -r -p "> " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    SHELL_PROFILE=""
    case "$SHELL" in
        */zsh)  SHELL_PROFILE="$HOME/.zshrc" ;;
        */bash) SHELL_PROFILE="$HOME/.bash_profile" ;;
        *)      SHELL_PROFILE="$HOME/.profile" ;;
    esac
    
    VENV_PATH=$(pwd)/.venv/bin/python3
    ALIAS_LINE="alias 1omni='$VENV_PATH -m omnitrain.cli'"
    
    if grep -q "alias 1omni=" "$SHELL_PROFILE"; then
        echo -e "  ${YELLOW}Notice:${NC} Alias '1omni' already exists in $SHELL_PROFILE. Skipping."
    else
        echo "$ALIAS_LINE" >> "$SHELL_PROFILE"
        echo -e "  ${GREEN}Alias added to $SHELL_PROFILE.${NC}"
        echo -e "  Please run ${CYAN}source $SHELL_PROFILE${NC} or restart your terminal."
    fi
fi

echo -e "\n${dim}OmniTrain Team: \"Fuse Everything. Trust Nothing. Verify Formally.\"${NC}\n"
