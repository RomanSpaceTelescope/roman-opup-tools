# OPUP, SCF, and Visit Parser

Parses Roman Space Telescope OPUP, SCF, and visit files. Information from these files is output it in CSV format.

## Authors

#### Developers
- David Morgan

#### Contributors
- Maxime Rizzo

## Features
- Parse OPUP, SCF, and individual visit files
- Extract metadata for each science exposure
- Read from gzipped archives (OPUP and SCF)
- Output results to CSV files
- Option to separate guide window information into a separate CSV file

## Usage

```
python visit_parser.py [-h] [-opup OPUP_FILEPATH [OPUP_FILEPATH ...]]
                       [-scf SCF_FILEPATH [SCF_FILEPATH ...]]
                       [-visit VISIT_FILEPATH [VISIT_FILEPATH ...]]
                       [-odir OUTPUT_DIR] [--keep_GW]
```

### Example:

```
python visit_parser.py -opup <OPUP_file_path_1> <OPUP_file_path_2> ...

python visit_parser.py -scf <SCF_file_path_1> <SCF_file_path_2> ...

python visit_parser.py -visit <visit_file_path_1> <visit_file_path_2> ...

```


### Arguments:
- `-opup`, `--opup_filepath`: Path(s) to the OPUP file(s)
- `-scf`, `--scf_filepath`: Path(s) to the SCF file(s)
- `-visit`, `--visit_filepath`: Path(s) to the visit file(s)
- `-odir`, `--output_dir`: Output CSV file directory
- `--keep_GW`: Keep guide window information in the same output CSV (separated by default)

## Output
The script generates CSV files containing metadata extracted from the input files. By default, guide window information (detector coordinate locations and GSDS entries) is separated into a separate CSV file (unless `--keep_GW` is specified).

## Dependencies
- python >= 3.10
- pandas

