
# OPUP Visit Parser

A Python tool for parsing NASA Roman Space Telescope OPUP (Observation Plan Upload) files and generating interactive HTML reports with sky plots and visit file syntax highlighting.

## 🎯 Overview

The OPUP Visit Parser extracts observation data from compressed OPUP archives (`.tar.gz`), SCF (Spacecraft File) files, and individual visit files (`.vst`). It generates:

- **Interactive HTML tables** with exposure metadata
- **Sky plots** showing target positions and Sun avoidance zones
- **Syntax-highlighted visit files** with clickable links
- **Statistics dashboards** with instrument breakdowns
- **CSV exports** for further analysis

## ✨ Features

- 📦 **Multiple input formats**: OPUP archives, SCF files, or individual visit files
- 🌌 **Interactive sky visualization**: Auto-generated sky plots with target positions
- 🎨 **STOL syntax highlighting**: Color-coded visit file display with VS Code dark theme
- 📊 **Statistics dashboard**: Visit counts, exposure totals, duration summaries
- 🔗 **Clickable visit links**: View raw visit file contents directly from HTML table
- 📈 **Instrument breakdown**: Per-instrument statistics and duration calculations
- 🌞 **Sun avoidance zones**: Automatic Sun position calculation for visit dates
- 💾 **Flexible output**: CSV, HTML, or both formats

## 🚀 Installation

### Requirements

```bash
pip install pandas numpy argparse pathlib
```

### Optional Dependencies (for sky plotting)

The tool can generate sky plots if `roman_plotter.py` is present in the same directory.

## 📖 Usage

### Basic Command Line

```bash
# Parse OPUP file(s)
python visit_parser.py -opup path/to/opup_file.tar.gz

# Parse SCF file(s)
python visit_parser.py -scf path/to/scf_file.tar.gz

# Parse individual visit file(s)
python visit_parser.py -visit path/to/visit_file.vst

# Multiple inputs
python visit_parser.py -opup file1.tar.gz file2.tar.gz -scf scf1.tar.gz
```

### Command Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--opup_filepath` | `-opup` | Path(s) to OPUP file(s) | `[]` |
| `--scf_filepath` | `-scf` | Path(s) to SCF file(s) | `[]` |
| `--visit_filepath` | `-visit` | Path(s) to visit file(s) | `[]` |
| `--output_dir` | `-odir` | Output file directory | Same as input |
| `--keep_GW` | | Keep Guide Window information | `False` |
| `--format` | | Output format: `csv`, `html`, or `both` | `html` |

### Examples

```bash
# Generate both CSV and HTML outputs
python visit_parser.py -opup my_observations.tar.gz --format both

# Specify custom output directory
python visit_parser.py -opup obs.tar.gz -odir ./results/

# Keep guide window data
python visit_parser.py -visit visit_001.vst --keep_GW

# HTML only (default)
python visit_parser.py -scf spacecraft_file.tar.gz
```

## 📂 Input File Formats

### OPUP Archive (`.tar.gz`)
Compressed archive containing:
- SCF files (spacecraft files)
- Visit files (`.vst`)
- Manifest files (`.man`)
- Observation definition files (`.json`)

### SCF Archive (`.tar.gz`)
Contains visit files and operation files:
- Visit files (`.vst`)
- Operation files (`.ops`)

### Visit Files (`.vst`)
STOL (Spacecraft Test and Operations Language) formatted files containing:
- Visit metadata
- Group/sequence/activity hierarchy
- Exposure commands and parameters

## 📊 Output Files

### HTML Report (`*_report.html`)
Interactive report featuring:
- Filterable/sortable data table
- Clickable visit file names
- Embedded visit file viewer with syntax highlighting
- Statistics dashboard
- Instrument breakdown
- Links to sky plotter

### Sky Plotter (`*_skyplot.html`)
Interactive celestial sphere visualization showing:
- Target positions (RA/Dec)
- Sun position and avoidance zones
- Visit footprints
- Ecliptic plane

### CSV Files
- `*_csv.csv` - Full exposure metadata
- `*_skyplot.csv` - Unique visit positions for plotting

## 🎨 Syntax Highlighting

The tool provides VS Code dark theme-style syntax highlighting for STOL visit files:

- **Keywords**: `VISIT`, `GROUP`, `SEQ`, `ACT` (purple/pink)
- **Visit IDs**: Highlighted identifiers (gold)
- **Commands**: Function names (yellow)
- **Parameters**: Key-value pairs (cyan/orange)
- **Numbers**: Numeric values (light green)
- **Strings**: Quoted strings (orange)
- **Comments**: `;@` prefixed lines (green italic)

## 📋 Parsed Metadata Fields

The tool extracts and organizes the following exposure metadata:

| Field | Description |
|-------|-------------|
| `Visit_ID` | Unique visit identifier |
| `SCI_ID` | Science exposure ID |
| `Visit_File_Name` | Source visit file |
| `Science_Instrument` | WFI, etc. |
| `EXPTIME` | Exposure duration (seconds) |
| `FILTER` | Optical filter used |
| `RA` | Right Ascension (degrees) |
| `DEC` | Declination (degrees) |
| `Duration` | Total visit duration |
| `GROUP`, `SEQUENCE`, `ACTIVITY` | Hierarchical indices |

Plus many more instrument-specific parameters extracted from visit files.

## 🔧 Key Functions

### Parsing Functions
- `parse_OPUP()` - Parse entire OPUP archive
- `parse_SCF()` - Parse SCF file
- `parse_visit_file()` - Parse individual visit file
- `parse_visit_lines()` - Parse STOL syntax

### HTML Generation
- `generate_html_report()` - Create interactive HTML table
- `syntax_highlight_visit_content()` - Apply STOL syntax highlighting
- `export_unique_visits_for_plotter()` - Generate sky plotter data

### Archive Utilities
- `get_SCF_from_OPUP()` - Extract SCF files from OPUP
- `get_visits_from_SCF()` - Extract visit files from SCF
- `get_visit_content()` - Read visit file from archive

## 🌐 Integration with Sky Plotter

When `roman_plotter.py` is available, the tool automatically:
1. Extracts unique visit positions (RA/Dec)
2. Calculates Sun position for observation date
3. Generates interactive sky plot with:
   - Target positions
   - Sun avoidance zones
   - Clickable visit markers
4. Cross-links between data table and sky plot

## 🐛 Troubleshooting

**Issue**: No columns returned for visit file  
**Solution**: Check that visit file contains valid STOL syntax with exposure commands

**Issue**: Sky plotter not generated  
**Solution**: Ensure `roman_plotter.py` is in the same directory

**Issue**: Visit file content not displaying  
**Solution**: Verify OPUP archive structure and visit file paths

## 📝 Notes

- Designed for NASA Roman Space Telescope observation planning files
- Handles nested `.tar.gz` archives automatically
- Supports both day-of-year and standard date formats
- Automatically prioritizes important columns in output
- Calculates visit durations and instrument statistics

## 🤝 Contributing

This tool is designed for NASA Roman Space Telescope operations. For enhancements or bug reports, contact your mission operations team.

## 📄 License

NASA Open Source - Internal Use

---

**Last Updated**: 2026-03-31  
**Compatible With**: Roman OPUP format specification v1.x
```

You can copy this entire markdown content and save it as `README.md` in your git repository. Simply:

1. Create a new file named `README.md` in your repository root
2. Copy and paste the entire content above
3. Save the file

The markdown will render beautifully on GitHub, GitLab, or any other git hosting platform with proper formatting, tables, code blocks, and emoji support!