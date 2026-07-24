# OPUP Report Generator

A Python tool for parsing NASA Roman Space Telescope OPUP (Observation Plan Upload) files and generating integrated HTML reports with sky plots, Gantt charts, visit file syntax highlighting, and aggregated CSV exports.

## 🎯 Overview

The OPUP Report Generator extracts observation data from compressed OPUP archives (`.tgz`), SCF (Spacecraft File) files, and individual visit files (`.vst`). It generates:

- **Integrated HTML reports** with exposure metadata, sky plots, and visit file viewer
- **Sky plots** showing target positions and Sun avoidance zones
- **Gantt charts** for multi-OPUP scheduling visualization
- **Syntax-highlighted visit files** with clickable links
- **Statistics dashboards** with instrument breakdowns
- **CSV exports** for further analysis
- **Aggregated outputs** when processing multiple OPUPs

## ✨ Features

- 📦 **Multiple input formats**: OPUP archives, SCF files, or individual visit files
- 📁 **Directory batch processing**: Recursively find and process all OPUPs in a folder
- 🌌 **Interactive sky visualization**: Auto-generated sky plots with target positions
- 📅 **Gantt chart generation**: Timeline visualization of observation schedules
- 🎨 **STOL syntax highlighting**: Color-coded visit file display with VS Code dark theme
- 📊 **Statistics dashboard**: Visit counts, exposure totals, duration summaries
- 🔗 **Clickable visit links**: View raw visit file contents directly from HTML table
- 📈 **Instrument breakdown**: Per-instrument statistics and duration calculations
- 🌞 **Sun avoidance zones**: Automatic Sun position calculation for visit dates
- 💾 **Flexible output**: Integrated (default), CSV, HTML, or both formats
- 📸 **Optional PNG sky plots**: Generate static sky plot images via `roman_visit_viewer`

## 🚀 Installation

### Prerequisites

1. Navigate to the root of the repo and install the package:
   ```bash
   pip install -e .
   ```

2. **Configure the ephemeris file**: The tool requires an OEM ephemeris file for accurate Sun position calculations. Create or update `config.json` in the repo root:
   ```json
   {
     "ephemeris": {
       "filename": "/path/to/ephem_file.oem",
       "description": "Ephemeris file for Roman mission observations"
     }
   }
   ```
   - The `filename` can be an absolute path or relative to the repo root
   - If `config.json` is missing, the tool falls back to looking for `RST_EPH_PRED_LONG_2026250_2027065_01.oem` in the repo root
   - **Tip:** Add your custom `config.json` to `.gitignore` if the ephemeris file path contains environment-specific data

## 📖 Usage

### Basic Command Line

```bash
# Basic integrated report for a single OPUP (default mode)
opup-report -opup my_observation_opup.tgz

# Process a single OPUP with custom output directory
opup-report -opup my_observation_opup.tgz -odir ./reports/

# Process multiple OPUP files
opup-report -opup opup_001.tgz opup_002.tgz opup_003.tgz

# Batch process all OPUPs in a directory (with aggregation + Gantt)
opup-report -opup_dir /path/to/opup_folder/ -odir ./output/

# Generate only CSV output (legacy mode)
opup-report -opup my_opup.tgz --format csv

# Generate both CSV and HTML (legacy mode)
opup-report -opup my_opup.tgz --format both

# Include Guide Window columns in CSV output
opup-report -opup my_opup.tgz --keep_GW

# Generate sky plot PNGs (slower) in addition to integrated report
opup-report -opup my_opup.tgz --pngs

# Process SCF and visit files alongside OPUPs (legacy mode)
opup-report -opup my_opup.tgz -scf SCF_001.scf -visit V01001001001.vis --format csv

# Generate Gantt chart from a previously-created aggregated CSV
opup-report --gantt aggregated_opups_20260428_143022.csv -odir ./charts/

# Directory mode with CSV-only output
opup-report -opup_dir /path/to/opup_folder/ --format csv

# Omit visit file links and embedded content (much smaller HTML for large OPUPs)
opup-report -opup my_opup.tgz --no-visit-links
```

### Command Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--opup_filepath` | `-opup` | Path(s) to OPUP `.tgz` archive file(s) | `[]` |
| `--opup_directory` | `-opup_dir` | Directory containing OPUP `.tgz` archives (recursive) | `None` |
| `--scf_filepath` | `-scf` | Path(s) to SCF file(s) (used in csv/html/both modes) | `[]` |
| `--visit_filepath` | `-visit` | Path(s) to visit file(s) (used in csv/html/both modes) | `[]` |
| `--output_dir` | `-odir` | Output directory | Same as input |
| `--keep_GW` | | Keep Guide Window columns in CSV output (default: separated to `_GWInfo.csv`) | `False` |
| `--pngs` | | Generate sky plot PNGs via `roman_visit_viewer` (slower) | `False` |
| `--gantt` | | Generate Gantt chart directly from an aggregated CSV file (skips OPUP parsing) | `None` |
| `--format` | | Output format: `integrated`, `csv`, `html`, or `both` | `integrated` |
| `--no-visit-links` | | Omit clickable visit file links and embedded visit content from the HTML report (significantly reduces file size for large OPUPs) | `False` |

### Output Formats

| Format | Description |
|--------|-------------|
| `integrated` | Full HTML report + interactive sky plotter + CSV + output archive per OPUP **(default)** |
| `csv` | CSV output only for OPUP, SCF, and visit files |
| `html` | Standalone HTML report only for OPUP files |
| `both` | Both CSV and HTML outputs |

## 📂 Input File Formats

### OPUP Archive (`.tgz`)

Compressed archive containing:
- SCF files (spacecraft files)
- Visit files (`.vis`)
- Manifest files (`.man`)
- Observation definition files (`.json`)

### SCF Files (`.scf`)

Contains visit files and operation files:
- Visit files (`.vis`)
- Operation files (`.ops`)

### Visit Files (`.vis`)

STOL (Spacecraft Test and Operations Language) formatted files containing:
- Visit metadata
- Group/sequence/activity hierarchy
- Exposure commands and parameters

## 📊 Output Files

### Integrated Report (default mode)

For each OPUP processed:
- Interactive HTML report with filterable/sortable data table
- Embedded visit file viewer with syntax highlighting
- Interactive sky plotter (HTML)
- CSV export with full exposure metadata
- Statistics dashboard with instrument breakdown
- Output archive (`.tgz`) bundling all deliverables

For directory/multi-OPUP mode:
- All individual OPUP outputs above
- Aggregated CSV across all OPUPs
- Gantt chart visualization

### Sky Plotter (`*_skyplot.html`)

Interactive celestial sphere visualization showing:
- Target positions (RA/Dec)
- Sun position and avoidance zones
- Visit footprints
- Ecliptic plane

### CSV Files

- `*_csv.csv` - Full exposure metadata
- `*_GWInfo.csv` - Guide Window information (separated by default)
- `*_skyplot.csv` - Unique visit positions for plotting
- `aggregated_opups_*.csv` - Combined data from multi-OPUP runs

### Gantt Chart

Timeline visualization showing:
- Observation scheduling across multiple OPUPs
- Visit durations and overlaps
- Generated from aggregated CSV data

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

### Visualization

- `generate_gantt_chart()` - Create Gantt chart from aggregated data
- Sky plot generation via `roman_plotter.py` / `roman_visit_viewer`

## 🌐 Integration with Sky Plotter

When `roman_plotter.py` is available, the tool automatically:
1. Extracts unique visit positions (RA/Dec)
2. Calculates Sun position for observation date
3. Generates interactive sky plot with:
   - Target positions
   - Sun avoidance zones
   - Clickable visit markers
4. Cross-links between data table and sky plot

When `--pngs` is specified and `roman_visit_viewer` is available, static PNG sky plots are also generated.

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| OEM ephemeris file not found warning | Create or update `config.json` in the repo root with the correct ephemeris file path (see Installation section) |
| No columns returned for visit file | Check that visit file contains valid STOL syntax with exposure commands |
| Sky plotter not generated | Ensure `roman_plotter.py` is in the same directory |
| Visit file content not displaying | Verify OPUP archive structure and visit file paths |
| Gantt chart not generated | Ensure aggregated CSV has required columns (Visit_ID, start/end times) |
| `--pngs` not producing output | Ensure `roman_visit_viewer` is installed and accessible |

## 📝 Notes

- Designed for NASA Roman Space Telescope observation planning files
- Handles nested `.tgz` archives automatically
- Supports both day-of-year and standard date formats
- Automatically prioritizes important columns in output
- Calculates visit durations and instrument statistics
- Guide Window data is separated to `_GWInfo.csv` by default (use `--keep_GW` to include)
- Directory mode (`-opup_dir`) recursively discovers all `.tgz` files
- The `--gantt` flag allows regenerating Gantt charts without re-parsing OPUPs

## 🤝 Contributing

This tool is designed for NASA Roman Space Telescope operations. For enhancements or bug reports, contact your mission operations team.

## 📄 License

NASA Open Source - Internal Use

---

*Last Updated: 2026-05-07*


