#!/bin/bash
# Export STL files for all Rover 3D printed parts
# Requires OpenSCAD command line tool

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if OpenSCAD is installed
if ! command -v openscad &> /dev/null; then
    echo "OpenSCAD is not installed. Please install it first."
    echo "brew install --cask openscad"
    exit 1
fi

# Base directory
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$BASE_DIR/models"

echo -e "${GREEN}=== The Rover STL Export Tool ===${NC}"
echo ""

# Function to export STL from SCAD
export_stl() {
    local scad_file=$1
    local part_function=$2
    local output_name=$3
    local output_dir=$4
    
    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"
    
    # Create temporary SCAD file with specific part
    temp_file="$output_dir/temp_export.scad"
    echo "include <$scad_file>;" > "$temp_file"
    echo "${part_function}();" >> "$temp_file"
    
    echo -e "${YELLOW}Exporting${NC} $output_name..."
    
    # Export to STL
    openscad -o "$output_dir/$output_name.stl" "$temp_file" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Exported $output_name.stl"
    else
        echo "✗ Failed to export $output_name"
    fi
    
    # Clean up temp file
    rm "$temp_file"
}

# =============================================
# TENNIS ROVER PARTS
# =============================================
echo -e "${GREEN}Tennis Rover Parts:${NC}"
echo "-------------------"

TENNIS_DIR="$MODELS_DIR/tennis-rover"
TENNIS_OUTPUT="$TENNIS_DIR/stl"

# Prototype parts
export_stl "$TENNIS_DIR/prototype-detailed.scad" "chassis_base" "prototype-chassis" "$TENNIS_OUTPUT"
export_stl "$TENNIS_DIR/prototype-detailed.scad" "collection_scoop" "prototype-scoop" "$TENNIS_OUTPUT"
export_stl "$TENNIS_DIR/prototype-detailed.scad" "wheel_assembly" "prototype-wheel" "$TENNIS_OUTPUT"
export_stl "$TENNIS_DIR/prototype-detailed.scad" "mini_basket" "prototype-basket" "$TENNIS_OUTPUT"
export_stl "$TENNIS_DIR/prototype-detailed.scad" "top_cover" "prototype-top-cover" "$TENNIS_OUTPUT"
export_stl "$TENNIS_DIR/prototype-detailed.scad" "elevator_mechanism" "prototype-elevator" "$TENNIS_OUTPUT"

# Tennis court accessories
export_stl "$TENNIS_DIR/mini-tennis-court-environment.scad" "mini_tennis_ball" "mini-tennis-ball" "$TENNIS_OUTPUT"
export_stl "$TENNIS_DIR/mini-tennis-court-environment.scad" "net_post" "net-post" "$TENNIS_OUTPUT"
export_stl "$TENNIS_DIR/mini-tennis-court-environment.scad" "fence_section" "fence-section" "$TENNIS_OUTPUT"
export_stl "$TENNIS_DIR/mini-tennis-court-environment.scad" "ball_hopper" "ball-hopper" "$TENNIS_OUTPUT"

# Full-size printable parts
if [ -f "$TENNIS_DIR/tennis-robot-printable-parts.scad" ]; then
    export_stl "$TENNIS_DIR/tennis-robot-printable-parts.scad" "part_front_chassis" "front-chassis" "$TENNIS_OUTPUT"
    export_stl "$TENNIS_DIR/tennis-robot-printable-parts.scad" "part_rear_chassis" "rear-chassis" "$TENNIS_OUTPUT"
    export_stl "$TENNIS_DIR/tennis-robot-printable-parts.scad" "part_ball_scoop" "ball-scoop" "$TENNIS_OUTPUT"
    export_stl "$TENNIS_DIR/tennis-robot-printable-parts.scad" "part_wheel_hub" "wheel-hub" "$TENNIS_OUTPUT"
    export_stl "$TENNIS_DIR/tennis-robot-printable-parts.scad" "part_wheel_tire" "wheel-tire" "$TENNIS_OUTPUT"
    export_stl "$TENNIS_DIR/tennis-robot-printable-parts.scad" "part_basket_small" "basket-small" "$TENNIS_OUTPUT"
    export_stl "$TENNIS_DIR/tennis-robot-printable-parts.scad" "part_basket_medium" "basket-medium" "$TENNIS_OUTPUT"
    export_stl "$TENNIS_DIR/tennis-robot-printable-parts.scad" "part_basket_large" "basket-large" "$TENNIS_OUTPUT"
    export_stl "$TENNIS_DIR/tennis-robot-printable-parts.scad" "part_solar_mount" "solar-mount" "$TENNIS_OUTPUT"
    export_stl "$TENNIS_DIR/tennis-robot-printable-parts.scad" "part_battery_housing" "battery-housing" "$TENNIS_OUTPUT"
    export_stl "$TENNIS_DIR/tennis-robot-printable-parts.scad" "part_electronics_mount" "electronics-mount" "$TENNIS_OUTPUT"
    export_stl "$TENNIS_DIR/tennis-robot-printable-parts.scad" "part_camera_mount" "camera-mount" "$TENNIS_OUTPUT"
    export_stl "$TENNIS_DIR/tennis-robot-printable-parts.scad" "part_bumper_corner" "bumper-corner" "$TENNIS_OUTPUT"
fi

echo ""

# =============================================
# SECURITY ROVER PARTS
# =============================================
echo -e "${GREEN}Security Rover Parts:${NC}"
echo "---------------------"

SECURITY_DIR="$MODELS_DIR/security-rover"
SECURITY_OUTPUT="$SECURITY_DIR/stl"

if [ -f "$SECURITY_DIR/security-rover-tank-chassis.scad" ]; then
    export_stl "$SECURITY_DIR/security-rover-tank-chassis.scad" "armored_chassis" "tank-chassis" "$SECURITY_OUTPUT"
    export_stl "$SECURITY_DIR/security-rover-tank-chassis.scad" "track_assembly" "track-assembly" "$SECURITY_OUTPUT"
    export_stl "$SECURITY_DIR/security-rover-tank-chassis.scad" "sensor_turret" "sensor-turret" "$SECURITY_OUTPUT"
    export_stl "$SECURITY_DIR/security-rover-tank-chassis.scad" "front_armor_plate" "front-armor" "$SECURITY_OUTPUT"
fi

echo ""

# =============================================
# SUMMARY
# =============================================
echo -e "${GREEN}=== Export Complete ===${NC}"
echo ""
echo "STL files have been exported to:"
echo "  • $TENNIS_OUTPUT"
echo "  • $SECURITY_OUTPUT"
echo ""
echo "To slice for 3D printing:"
echo "  1. Open files in your slicer (Cura, PrusaSlicer, etc.)"
echo "  2. Use recommended settings from documentation"
echo "  3. Check docs/ folder for material and infill guidelines"
echo ""
echo -e "${GREEN}Happy printing!${NC} 🤖"