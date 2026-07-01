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

echo -e "\n${BLUE}[4/4]${NC} Installation complete."


echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}  OMNITRAIN READY FOR INDUSTRIAL DEPLOYMENT${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "\nUsage Guide:"
echo -e "  Import OmniTrain in Python scripts to scaffold projects, train networks, or run models:"
echo -e "  ${CYAN}import omnitrain as ot${NC}"


echo -e "\n${dim}OmniTrain Team: \"Fuse Everything. Trust Nothing. Verify Formally.\"${NC}\n"
