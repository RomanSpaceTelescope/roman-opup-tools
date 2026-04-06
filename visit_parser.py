# Authors:
# David Morgan
# Maxime Rizzo

#%%
import os
from pathlib import Path
import io
import re
from ast import literal_eval
import pandas as pd
import tarfile
import zipfile
import argparse
import html as html_module
import json

# Columns to make first in the output CSV file (if available)
PRIORITY_COLUMNS = ['Visit_ID', 'SCI_ID']

#%%

def parse_visit_header(visit_header_line):

    visit_info = {}
    items = [x.strip() for x in visit_header_line.split(',')]

    visit_info['Visit_ID'] = items[1].replace('V','') if items[1].startswith('V') and items[1][1:].isdigit() else None

    for item in items:
        if '=' in item:
            key, val = item.split('=')
            visit_info[key] = val
    
    return visit_info


def extract_exposure_metadata(parsed_visit):
    '''
    This function takes the parsed visit file output and interprets the metadata
    corresponding to each science exposure.
    
    PARAMS:
    -------
    parsed_visit - information extracted from the visit file using the parse_visit_file function

    RETURNS:
    -------
    Pandas dataframe with metadata info for each science exposure.

    '''
    
    # Initialize empty dict to store snapshot of current metadata:
    current_metadata = {}

    # Initialize visit info for the given visit
    visit_info = parsed_visit.pop('visit')

    # Initalize list of exposure metadata
    exposures = []

    # Iterate through each group in the visit
    for group_num, group in parsed_visit.items():
        if isinstance(group, dict):
            # Iterate each sequence within the group
            for seq_num in group['sequences']:
                sequence = group[seq_num]

                # Interate each activity within the sequence
                for act_num in sequence['activities']:
                    activity = sequence[act_num]
                    
                    # Grab the parameters from the current activity's command
                    # params = parsed_visit[group_num][seq_num][act_num]['command']
                    params = activity['command']
        
                    # Update current metadata parameters
                    current_metadata.update(params)
                    
                    # Check if this activity is the start of an exposure
                    if params['command'] in ['WFI_EXPOSURE_START_F', 'WFI_SET_USER_ID_SCI_EXPOSE_F']:
        
                        # Save group, seq, and act numbers to current metadata
                        current_metadata.update({
                                                'GROUP': group_num,
                                                'SEQUENCE': seq_num,
                                                'ACTIVITY': act_num})

                        # Add visit ID, early start, late start, and late end times
                        current_metadata.update(visit_info)

                        # Save target type if it exists:
                        if 'TARGET_TYPE' in parsed_visit.keys():    
                            current_metadata.update({'TARGET_TYPE': parsed_visit['TARGET_TYPE']})
        
                        # Save all current metadata parameters in exposure list
                        # as a copy of the dict to avoid overwriting data.
                        exposures.append(current_metadata.copy())


    # Convert list of exposures to pandas dataframe
    exp_df = pd.DataFrame(exposures)
    
    # # Setting the science ID to be the dataframe index
    # if 'SCI_ID' in exp_df.columns:
    #     exp_df.set_index('SCI_ID', inplace=True)
    # elif 'SCI_USER_ID' in exp_df.columns:
    #     exp_df.set_index('SCI_USER_ID', inplace=True)
    
    return exp_df

def parse_visit_file_obj(file_obj):
    '''
    This function parses a visit file object and returns a nested dictionary
    containing metadata info for the groups, sequences, and activities
    defined in the visit.

    PARAMS:
    -------
    file_obj - file object of the visit file


    RETURNS:
    -------
    visit - dict, nested dictionary containing metadata info. Groups,
            sequences, and activities are separated by their numbers.
            e.g. visit[1] would return info for the first group, visit[1][1]
            would return info for the first sequence of the first group, and
            visit[1][1][1] would return info for the first activity of the
            first sequence of the first group, all in the form of a dictionary.
    '''
    
    # Initializing dict to store visit info
    visit = {}

    # Iterate by line in the visit file
    for line in file_obj:
        
        
        # Remove trailing white space
        line = line.strip()

        if 'Target Type' in line:
            target_type_match = re.match(r'.* Target Type = (.+)', line)
            if target_type_match:
                visit['TARGET_TYPE']=target_type_match.group(1)
        
        # Parse VISIT line
        if line.startswith('VISIT'):
            visit['visit'] = parse_visit_header(line)

            # Initialize list of group number as empty list
            visit['groups'] = []

        ### Parse GROUP line
        elif line.startswith('GROUP'):
            group_match = re.match(r'GROUP, (\d+), CONGRP=(.*);', line)
            if group_match:
                # Extracting the group number
                group_num = int(group_match.group(1))

                # Creating dict for the extracted group info
                current_group = {'group_num': group_num, 'congrp': group_match.group(2), 'sequences': []}

                # Adding group number to list of groups
                visit['groups'].append(group_num)

                # Adding current group to visit
                visit[group_num] = current_group

        ### Parse SEQ line
        elif line.startswith('SEQ'):
            seq_match = re.match(r'SEQ, (\d+), CONSEQ=(.*);', line)
            if seq_match:
                # Extracting the sequence number
                seq_num = int(seq_match.group(1))

                # Creating dict for the extracted sequence info
                current_seq = {'seq_num': seq_num, 'conseq': seq_match.group(2), 'activities': []}

                # Adding sequence number to list of sequences
                visit[group_num]['sequences'].append(seq_num)


                # Adding current sequence to group
                visit[group_num][seq_num] = current_seq

        ### Parse ACT line
        elif line.startswith('ACT'):
            # Regex match for the activity definition
            act_match = re.match(r'ACT,\s+(\d+),(.*)', line)
            if act_match:
                # Extracting the activity number
                act_num = int(act_match.group(1))

                # Creating dict for the extracted activity info
                current_act = {
                    'act_num': act_num,
                    'command': parse_visit_command(act_match.group(2))
                }

                # Adding new activity number to activity list:
                visit[group_num][seq_num]['activities'].append(act_num)

                # Adding activity 
                visit[group_num][seq_num][act_num] = current_act


    
    # returning visit information
    return visit


def parse_visit_file(filepath):
    '''
    This function parses a visit file and returns a nested dictionary
    containing metadata info for the groups, sequences, and activities
    defined in the visit.

    PARAMS:
    -------
    filepath - str, path to the visit file


    RETURNS:
    -------
    visit - dict, nested dictionary containing metadata info. Groups,
            sequences, and activities are separated by their numbers.
            e.g. visit[1] would return info for the first group, visit[1][1]
            would return info for the first sequence of the first group, and
            visit[1][1][1] would return info for the first activity of the
            first sequence of the first group, all in the form of a dictionary.
    '''
    
    try:
        # Open the visit file for parsing:
        visit_file_obj = read_visit_file(filepath)

        # Parsing the visit file object
        visit = parse_visit_file_obj(visit_file_obj)
        
        # Converting to data frame
        visit_df = extract_exposure_metadata(visit)

        # returning visit information
        return visit_df
    except Exception as e:
        # import traceback
        # traceback.print_exc()
        print(f'Encountered error while processing visit {filepath}: {e}')
        return pd.DataFrame()



def parse_visit_command(command_str):
    '''
    This function parses a given activity command and returns a dictionary of command
    parameters.

    PARAMS:
    command_str - str, command as it is written in the visit file


    RETURNS:
    command - dict, dictionary of metadata info for the given command
    '''

    try:
        
        # Remove trailing white space in command
        command_str = command_str.strip()
    
        # Initialize empty dictionary with command info
        command = {}
    
        if ';' in command_str:
            # Splitting command by semicolon
            command_str_split = [x for x in command_str.split(';') if len(x)>0]
            
            if len(command_str_split)==2:
                command_function = command_str_split[0]
                command_comments = command_str_split[1]
            else:
                # If there are multiple semicolons on the line, it means that
                # a command has likely been temporarily commented out and 
                # replace with a wait, in which case we only save the function.
                command_function = command_str_split[0]
                command_comments = ''
            
            
        else:
            command_function = command_str
            command_comments = ''
    
        # Getting the parameter names if available:
        cmd_param_names = command_comments.strip().split(',')
    
        # Regex match for the activity definition
        cmd_match = re.match(r'(\w+)\((.+)\)', command_function.strip())

        ### Extracting command info if the regex could be matched.
        if cmd_match:
            # Extracting command info
            cmd_name = cmd_match.group(1)
            cmd_params = cmd_match.group(2).strip().split(',')
    
            # Adding command name to the dict
            command['command']=cmd_name
    
            # There are parameters passed in the function call
            params_present=True
            
            # If the parameter names could be found
            if (cmd_param_names[0]!='') and (len(cmd_params)==len(cmd_param_names)):
                param_names_present=True
            else:
                param_names_present=False
    
        elif '(' not in command_str:
            # If no parameters are passed in the command, only return the command name
            command['command'] = command_str.replace(';', '')
    
            # No parameters are present
            params_present=False
            param_names_present=False
    
        else:
            # If the format could not be recognized, return the command as written
            command['command'] = command_str
    
            # Assume no parameters to parse
            params_present=False
            param_names_present=False
    
    
        # Adding the parameter names and parameters to the dict if available.
        # (If either are unavailable, do nothing).
        if params_present and param_names_present:
            # If the command has parameters and the names of the 
            # parameters are included after the semicolon, we can
            # use the parameter names in the metadata.
    
            # Checking that the parameter names and parameters match
            if len(cmd_params)==len(cmd_param_names):
                # Parsing command parameters and parameter names
                for key, val in zip(cmd_param_names, cmd_params):
                    
                    try:
                        command[key.strip()] = literal_eval(val.strip())
                    except ValueError:
                        command[key.strip()] = val.strip()
                    except SyntaxError:
                        command[key.strip()] = val.strip()
                        
                # This dict contains information for special commands that are usually executed multiple
                # times prior to a science exposure. Since the parameters will be overwritten for each
                # consecutive run, we need to rename the parameters to be unique.
                specials = {
                            'WFI_SRCS':{'key':'BANK', 'desc':'BANK'},
                            'FGS_GSDS_ENTRY':{'key':'WFI_DET', 'desc':'GW'},
                            'SCE_GW_CONFIG_LOC':{'key':'SCENUM','desc':'GW'}
                            }

                # Checking to see if the command is one of the special commands requiring the parameters to be
                # renamed to avoid being overwritten.
                for spec_cmd in specials:
                    if spec_cmd in cmd_name:
                        command = rename_command_params(command, specials[spec_cmd]['key'], specials[spec_cmd]['desc'])
                        command['command'] = command_str

        elif params_present:
            # If the command has parameters, but the names of the parameters
            # could not be found, only pass the parameters.
    
            for key, val in zip([f'PARAM{n:003}' for n in range(1,len(cmd_params))], cmd_params):
                try:
                    command[key] = literal_eval(val)
                except ValueError:
                    command[key] = val

        
        # Returning command dict
        return command
    except:
        print(f'Could not parse command: {command_str}')
        import traceback
        traceback.print_exc()
        return None

def rename_command_params(command, key, desc):
    to_add = command[key]
    command = {f'{x}_{desc}{to_add:02}':y for x,y in command.items() if x not in [key, 'command']}
    return command
    

def get_SCF_from_OPUP(opup_filepath):

    with tarfile.open(opup_filepath, 'r:gz') as tarball_file:
        
        # Search the tarball members for the given visit ID:
        scf_files = [os.path.join(opup_filepath, x) for x in tarball_file.getnames() if x.startswith('SCF')]
    
    return scf_files

def get_manifest_from_OPUP(opup_filepath):

    with tarfile.open(opup_filepath, 'r:gz') as tarball_file:
        
        # Search the tarball members for the given visit ID:
        manifest_filepaths = [os.path.join(opup_filepath, x) for x in tarball_file.getnames() if x.endswith('.man')]

    return manifest_filepaths

def get_odf_from_OPUP(opup_filepath):

    with tarfile.open(opup_filepath, 'r:gz') as tarball_file:
        
        # Search the tarball members for the given visit ID:
        odf_filepaths = [os.path.join(opup_filepath, x) for x in tarball_file.getnames() if 'odf' in x and x.endswith('.json')]

    return odf_filepaths

def get_visits_from_SCF(scf_filepath):
    
    # If the SCF is within a gzipped OPUP
    if not Path(scf_filepath).parent.is_dir() and tarfile.is_tarfile(Path(scf_filepath).parent.as_posix()):
        with tarfile.open(os.path.dirname(scf_filepath), 'r:gz') as tarball_file:
            # Extracting file as object
            scf_file = tarball_file.extractfile(os.path.basename(scf_filepath))

            # Accessing the gzipped scf file object within the gzipped opup
            with tarfile.open(fileobj=scf_file, mode='r:gz') as tarball_file:
                
                # Search the tarball members for visit files:
                visit_files = [os.path.join(scf_filepath, x) for x in tarball_file.getnames() if x.endswith('.vst')]
    elif tarfile.is_tarfile(scf_filepath):
        # Accessing the gzipped scf file object
        with tarfile.open(scf_filepath, mode='r:gz') as tarball_file:
            
            # Search the tarball members for visit files:
            visit_files = [os.path.join(scf_filepath, x) for x in tarball_file.getnames() if x.endswith('.vst')]
    else:
        visit_files = [os.path.join(scf_filepath, x) for x in os.listdir(scf_filepath) if x.endswith('.vst')]
    
    return visit_files

def get_ops_from_SCF(scf_filepath):

    with tarfile.open(scf_filepath, 'r:gz') as tarball_file:
        
        # Search the tarball members for the given visit ID:
        ops_files = [os.path.join(scf_filepath, x) for x in tarball_file.getnames() if x.endswith('.ops')]

    return ops_files

def parse_opup_manifest(opup_filepath):

    # Getting the filepath of the manifest file
    manifest_filepaths = get_manifest_from_OPUP(opup_filepath)
    
    # Parsing each manifest and returning the info as a list of files
    manifest_info = []
    for manifest_filepath in manifest_filepaths:
        manifest_info.extend(parse_manifest(manifest_filepath))
    
    return manifest_info

def parse_manifest(manifest_filepath):

    # Extracting manifest output
    with tarfile.open(os.path.dirname(manifest_filepath), 'r:gz') as tarball_folder:
        f = tarball_folder.extractfile(os.path.basename(manifest_filepath))

        # Reading reach line and decoding
        manifest_lines = [x.decode() for x in f]
        manifest_output = [line.strip() for line in manifest_lines if line and not line.startswith('#')]

    return manifest_output


def read_visit_file(visit_filepath):
    """This function accepts a visit file path and returns a StringIO object containing the visit file text.
    The file path can be within gzipped SCF and OPUP archives, or not.

    Args:
        visit_filepath (str): Path to the visit file.

    Returns:
        StringIO: StringIO stream containing the visit file text.
    """

    
    ### Identifying if the visit file path is contained in an SCF and OPUP directory
    if Path(visit_filepath).parent.stem.startswith('SCF'):
        scf_filepath = Path(visit_filepath).parent.as_posix()
    else:
        scf_filepath = ''

    if Path(scf_filepath).parent.stem.endswith('opup'):
        # opup_filepath = os.path.dirname(scf_filepath)
        opup_filepath = Path(scf_filepath).parent.as_posix()
    else:
        opup_filepath = ''


    ### Identifying if those directories are gzipped

    # Determine if the opup is gzipped
    if opup_filepath !='':
        opup_gzipped = tarfile.is_tarfile(opup_filepath)
    else:
        opup_gzipped = False
    
    if opup_gzipped:
        # Opening the opup tarball
        opup_tarball = tarfile.open(opup_filepath, 'r:gz')
        # opup_tarball = io.BytesIO(tarfile.open(opup_filepath, 'r:gz'))

        # Extracting SCF file
        scf_file = opup_tarball.extractfile(Path(scf_filepath).name)
        # scf_file = io.BytesIO(opup_tarball.extractfile(Path(scf_filepath).name).read())

        # Determine if the scf file object is gzipped too
        scf_gzipped = tarfile.is_tarfile(scf_file)
        scf_file.seek(0)
    else:
        if scf_filepath !='':
            # Determing if the scf file path is gzipped
            scf_gzipped = tarfile.is_tarfile(scf_filepath)
        else:
            scf_gzipped = False

    # print(visit_filepath)
    # print(opup_filepath)
    # print(scf_filepath)
    # print(opup_gzipped, scf_gzipped, tarfile.is_tarfile(scf_file))

    ### Extracting the info from the visit file and creating a StringIO object containing it.

    if scf_gzipped:
        if opup_gzipped:
            # This covers the case when both the OPUP and SCF are gzipped.
            # Accessing the gzipped scf file object within the gzipped opup
            with tarfile.open(fileobj=scf_file, mode='r:gz') as scf_tarball:
                # Extracting visit file
                visit_file_ex = scf_tarball.extractfile(Path(visit_filepath).name)

                # Getting visit file text
                visit_file_obj= io.StringIO(visit_file_ex.read().decode())

            # Closing scf tarball
            scf_file.close()

            # Closing opup tarball
            opup_tarball.close()

        else:
            # This covers the case when just the SCF is gzipped
            # If there is no gzipped OPUP folder, then we can access the gzipped SCF directly.
            with tarfile.open(scf_filepath, mode='r:gz') as scf_tarball:
                # Extracting visit file
                visit_file_ex = scf_tarball.extractfile(Path(visit_filepath).name)

                # Getting visit file text
                visit_file_obj= io.StringIO(visit_file_ex.read().decode())
    else:
        # This covers the case if the visit file is not in a gzipped archive at all:
        with open(visit_filepath, 'r') as visit_file_ex:
            visit_file_obj = io.StringIO(visit_file_ex.read())

    

    return visit_file_obj

def process_odf_files(json_files):
    """
    Process a list of JSON files and extract visit data.

    Args:
    json_files (list): A list of file paths to JSON files.

    Returns:
    list: A list of pandas DataFrames, each containing visit data from a JSON file.
    """
    outputs = []
    for json_file in json_files:
        
        opup_filepath = Path(json_file).parent.as_posix()
        if tarfile.is_tarfile(opup_filepath):
            # If the json file is in a gzipped tarball, we need to extract the file object
            with tarfile.open(opup_filepath, mode='r:gz') as opup_tarball:
                # Extracting json file
                json_file_ex = opup_tarball.extractfile(Path(json_file).name)

                # Getting json file text as a string IO
                json_file= io.StringIO(json_file_ex.read().decode())

        # Read the JSON file
        data = pd.read_json(json_file)

        # Extract the 'visits' data
        visits = data['visits']

        # Convert each visit to a pandas Series and create a DataFrame
        dataf = pd.DataFrame([pd.Series(visit) for visit in visits])

        # Add a column with the name of the parent directory (assumed to be 'opup')
        dataf['opup'] = os.path.basename(opup_filepath)

        # Append the DataFrame to the outputs list
    outputs.append(dataf)

    return outputs


def export_obsplan_from_json(json_files):
    """
    Export observation plan data from JSON files to an Excel file.

    Args:
    json_files (list): A list of file paths to JSON files containing observation plan data.

    Returns:
    pd.DataFrame: A DataFrame containing the combined observation plan data from all input JSON files.
    """
    outputs = process_odf_files(json_files)
    output = pd.concat(outputs, ignore_index=True)
    return output

def parse_OPUP(opup_filepath):

    # Getting a list of all visit file paths in the opup
    scf_files = get_SCF_from_OPUP(opup_filepath)

    # Parsing each SCF
    opup_info = pd.DataFrame()
    for scf_filepath in scf_files:
        scf_info = parse_SCF(scf_filepath)

        opup_info = pd.concat((opup_info, scf_info))

    # Parsing the obsplan json
    obsplan_filepaths = get_odf_from_OPUP(opup_filepath)
    obsplan_df= export_obsplan_from_json(obsplan_filepaths)

    # Merging the obsplan df with the opup df
    opup_info = pd.merge(obsplan_df, opup_info, on=['Visit_ID'])

    return opup_info
#%%
def parse_SCF(scf_filepath):
    try:
        
        # Get list of visit file paths from the given SCF
        visit_files = get_visits_from_SCF(scf_filepath)

        # Parsing each visit file and adding to a pandas df
        scf_info = pd.DataFrame()
        for visit_file in visit_files:

            # Parsing visit file
            visit_df = parse_visit_file(visit_file)

            # Concatenating df to opup info df
            scf_info = pd.concat((scf_info, visit_df))
        
    except Exception as e:
        print(f'Encountered error while processing {scf_filepath}: {e}')
        scf_info = pd.DataFrame()

    return scf_info


def get_current_gw_columns(df):
    """Returns a list of Guide Window related columns in the given data frame. This is useful for separating the GW columns.


    Args:
        df (pd.DataFrame): DataFrame containing the Guide Window columns to identify.

    Returns:
        list: List of guide window column names in the given DataFrame.
    """

    gw_columns=['ACQ_USE',
                'ACQ_H',
                'ACQ_V',
                'TRK_USE',
                'TRK_H',
                'TRK_V',
                'EDGE',
                'LOW',
                'NOM',
                'HIGH',
                'SKYBGND',
                'STAR_TEMP',
                'X_START',
                'Y_START']
    
    current_gw_cols = []
    
    for x in range(1,19):
        for col in gw_columns:
            test_col = f'{col}_GW{x:02}'
            if test_col in df.columns:
                    current_gw_cols.append(f'{col}_GW{x:02}')

    return current_gw_cols

def split_df_columns(df, columns_to_split):

    # Creating new DataFrame with the split columns
    df_2 = df[columns_to_split]

    # Dropping split columns from original DataFrame
    df_1 = df.drop(columns=columns_to_split)
    
    return df_1, df_2

def prioritize_columns(df, priority_columns):

    # Filtering out columns that are not available in the given DataFrame
    priotiry_cols = set(priority_columns) - set(df.columns)

    # Re-order columns in the data frame to prioritize the given columns
    new_order = priority_columns + [col for col in df.columns if col not in priority_columns]

    return df[new_order]

def find_nontgz_parent(folderpath):
    # Returning the first path in the parents of the given path
    # that is a directory.
    for path in Path(folderpath).parents:
        if Path(path).is_dir():
            return path

def write_to_CSV(df, output_csv, keep_GW=True):
    '''
    This function writes a given pandas DataFrame to a CSV file.
    If the dataframe has Guide Window information, this function will
    extract this information to a separate CSV.

    PARAMS:
    df - pandas DataFrame
    output_csv - str, path to the output CSV file
    keep_GW - bool, whether to keep Guide Window columns in the main CSV

    RETURNS:
    None
    '''
    # Convert Path to string if necessary
    output_csv = str(output_csv)
    
    # Getting guide window columns
    gw_cols = get_current_gw_columns(df)

    if len(gw_cols)>0:
        # Splitting the Guide Window info from the main dataframe
        gw_df, df_out = split_df_columns(df, gw_cols)

        # Writing Guide Window info to separate CSV
        gw_df.to_csv(output_csv.replace('.csv', '_GWInfo.csv'), index=False)

        if keep_GW:
            # Writing to CSV with GW info
            df.to_csv(output_csv, index=False)
        else:
            # Writing to CSV without GW info
            df_out.to_csv(output_csv, index=False)
    else:
        # Writing to CSV
        df.to_csv(output_csv, index=False)

def setup_parser():
    parser = argparse.ArgumentParser(description='Parse OPUP files.')
    parser.add_argument('-opup', '--opup_filepath', type=str, nargs='+', help='Path(s) to the OPUP file(s)', default=[])
    parser.add_argument('-scf', '--scf_filepath', type=str, nargs='+', help='Path(s) to the SCF file(s)', default=[])
    parser.add_argument('-visit', '--visit_filepath', type=str, nargs='+', help='Path(s) to the visit file(s)', default=[])
    parser.add_argument('-odir', '--output_dir', type=str, help='Output CSV file directory', default=None)
    parser.add_argument('--keep_GW', action='store_true', help='Keep Guide Window information in the output CSV.')
    return parser

def process_OPUPs(opup_filepaths, output_dir=None, keep_GW=True):

    # Iterating through list of opup filepaths
    for opup_filepath in opup_filepaths:
        # If the output directory is not given, then we use the local directory.
        # If the local directory is a file (i.e. gzipped archive), then we use the closest non-gzipped parent.
        if output_dir is None:
            output_dir = Path(opup_filepath).parent.as_posix()

            # If the SCF file is stored in a gzipped archive, we have to modify the save path to accommodate this.
            if not Path(output_dir).is_dir():
                output_dir = find_nontgz_parent(output_dir)

        output_csv = Path(output_dir).joinpath(Path(opup_filepath).stem + '_csv.csv').as_posix()

        # Parse the opup to get the opup info
        opup_info = parse_OPUP(opup_filepath)

        # Move priority columns to the left-side of the data frame
        opup_info = prioritize_columns(opup_info, PRIORITY_COLUMNS)

        # Write to CSV
        if len(opup_info.columns)>0:
            write_to_CSV(opup_info, output_csv, keep_GW=keep_GW)
        else:
            print(f'No columns were returned for {opup_filepath}')

def process_SCFs(scf_filepaths, output_dir=None, keep_GW=True):

    # Iterating through list of SCF file paths
    for scf_filepath in scf_filepaths:
        # If the output directory is not given, then we use the local directory.
        # If the local directory is a file (i.e. gzipped archive), then we use the closest non-gzipped parent.
        if output_dir is None:
            output_dir = Path(scf_filepath).parent.as_posix()

            # If the SCF file is stored in a gzipped archive, we have to modify the save path to accommodate this.
            if not Path(output_dir).is_dir():
                output_dir = find_nontgz_parent(output_dir)

        output_csv = Path(output_dir).joinpath(Path(scf_filepath).stem + '_csv.csv').as_posix()

        # Parsing SCF
        scf_info = parse_SCF(scf_filepath)

        # Move priority columns to the left-side of the data frame
        scf_info = prioritize_columns(scf_info, PRIORITY_COLUMNS)

        # Writing to CSV
        if len(scf_info.columns)>0:
            write_to_CSV(scf_info, output_csv, keep_GW=keep_GW)
        else:
            print(f'No columns were returned for {scf_filepath}')

def process_visits(visit_filepaths, output_dir=None, keep_GW=True):

    # Iterating by visit file path
    for visit_filepath in visit_filepaths:

        # If the output directory is not given, then we use the local directory.
        # If the local directory is gzipped, we move to the closest non-gzipped parent.
        if output_dir is None:
            output_dir = Path(visit_filepath).parent.as_posix()

            # If the visit file is stored in a gzipped archive, we have to modify the save path to accommodate this.
            if not Path(output_dir).is_dir():
                output_dir = find_nontgz_parent(output_dir)

        # Creating output csv file path
        output_csv = Path(output_dir).joinpath(Path(visit_filepath).stem + '_csv.csv').as_posix()

        # Parsing visit file
        visit_info = parse_visit_file(visit_filepath)

        # Move priority columns to the left-side of the data frame
        visit_info = prioritize_columns(visit_info, PRIORITY_COLUMNS)

        if len(visit_info.columns) > 0:
            write_to_CSV(visit_info, output_csv, keep_GW=keep_GW)
        else:
            print(f'No columns were returned for {visit_filepath}')

#%%

def export_opup_to_html(opup_filepath, output_html_path=None, keep_GW=True):
    """
    Export OPUP data to an HTML file with hyperlinks to individual visit files.
    
    Args:
        opup_filepath: Path to the OPUP .tgz archive
        output_html_path: Path for output HTML file (optional)
        keep_GW: Whether to keep Guide Window columns
    
    Returns:
        Path to generated HTML file
    """
    # Parse the OPUP (reuse your existing function)
    opup_info = parse_OPUP(opup_filepath)
    
    # Prioritize columns
    opup_info = prioritize_columns(opup_info, PRIORITY_COLUMNS)
    
    # Remove GW columns if requested
    if not keep_GW:
        gw_cols = get_current_gw_columns(opup_info)
        opup_info = opup_info.drop(columns=gw_cols)
    
    # Determine output path
    if output_html_path is None:
        output_dir = Path(opup_filepath).parent
        if not output_dir.is_dir():
            output_dir = find_nontgz_parent(opup_filepath)
        output_html_path = Path(output_dir) / f"{Path(opup_filepath).stem}_report.html"
    
    # Generate HTML
    html_content = generate_html_report(opup_info, opup_filepath)
    
    # Write HTML file
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {output_html_path}")
    return output_html_path

def get_all_visit_contents(opup_filepath, visit_filenames):
    """
    Extract contents of all visit files from the OPUP archive.
    Visit files are nested inside SCF tarballs within the OPUP.
    
    Args:
        opup_filepath: Path to the OPUP .tgz archive
        visit_filenames: List of visit filenames to extract
    
    Returns:
        Dictionary mapping visit_filename -> content
    """
    visit_contents = {}
    
    try:
        if tarfile.is_tarfile(opup_filepath):
            with tarfile.open(opup_filepath, 'r:gz') as opup_tar:
                # Find all SCF tarballs in the OPUP
                scf_members = [m for m in opup_tar.getmembers() if 'SCF' in m.name and m.name.endswith('.tgz')]
                
                print(f"Found {len(scf_members)} SCF tarball(s) in OPUP")
                
                # Process each SCF tarball
                for scf_member in scf_members:
                    print(f"  Processing {scf_member.name}...")
                    
                    # Extract the SCF tarball file object
                    scf_file_obj = opup_tar.extractfile(scf_member)
                    
                    if scf_file_obj:
                        # Open the nested SCF tarball
                        with tarfile.open(fileobj=scf_file_obj, mode='r:gz') as scf_tar:
                            # Look for visit files in this SCF
                            for member in scf_tar.getmembers():
                                filename = Path(member.name).name
                                
                                if filename in visit_filenames and filename not in visit_contents:
                                    # Extract the visit file content
                                    visit_file_obj = scf_tar.extractfile(member)
                                    if visit_file_obj:
                                        content = visit_file_obj.read().decode('utf-8', errors='replace')
                                        visit_contents[filename] = content
                                        print(f"    Extracted {filename} ({len(content)} chars)")
                
                # Mark any missing files
                for visit_filename in visit_filenames:
                    if visit_filename not in visit_contents:
                        visit_contents[visit_filename] = "Visit file not found in archive"
                        print(f"  Warning: {visit_filename} not found")
                        
    except Exception as e:
        print(f"Error reading archive {opup_filepath}: {e}")
        import traceback
        traceback.print_exc()
    
    return visit_contents

def extract_visit_file_contents(opup_filepath, visit_filename):
    """
    Extract the content of a specific visit file from the OPUP archive.
    Visit files are nested inside SCF tarballs within the OPUP.
    
    Args:
        opup_filepath: Path to the OPUP .tgz archive
        visit_filename: Name of the visit file to extract (e.g., 'V0010801001001006002.vst')
    
    Returns:
        String content of the visit file, or None if not found
    """
    try:
        if tarfile.is_tarfile(opup_filepath):
            with tarfile.open(opup_filepath, 'r:gz') as opup_tar:
                # Find all SCF tarballs in the OPUP
                scf_members = [m for m in opup_tar.getmembers() if 'SCF' in m.name and m.name.endswith('.tgz')]
                
                # Search each SCF tarball for the visit file
                for scf_member in scf_members:
                    scf_file_obj = opup_tar.extractfile(scf_member)
                    
                    if scf_file_obj:
                        # Open the nested SCF tarball
                        with tarfile.open(fileobj=scf_file_obj, mode='r:gz') as scf_tar:
                            # Look for the visit file
                            for member in scf_tar.getmembers():
                                if member.name.endswith(visit_filename):
                                    visit_file_obj = scf_tar.extractfile(member)
                                    if visit_file_obj:
                                        content = visit_file_obj.read().decode('utf-8', errors='replace')
                                        return content
        
        return None
        
    except Exception as e:
        print(f"Error extracting {visit_filename}: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_html_report(df, opup_filepath, sky_plotter_html=None):
    """
    Generate HTML content with hyperlinks to visit files and horizontal scrolling.
    Includes embedded visit file contents with syntax highlighting.
    
    Args:
        df: DataFrame containing OPUP data
        opup_filepath: Path to the OPUP archive for creating links
        sky_plotter_html: Optional path to sky plotter HTML file for cross-linking
    
    Returns:
        HTML string
    """
    opup_name = Path(opup_filepath).name
    
    # Extract all visit file contents
    print("Extracting visit file contents from archive...")
    visit_filenames = df['Visit_File_Name'].dropna().unique().tolist()
    visit_contents = get_all_visit_contents(opup_filepath, visit_filenames)
    print(f"Extracted {len(visit_contents)} visit files")
    
    # Apply syntax highlighting to each visit file
    print("Applying syntax highlighting...")
    visit_contents_highlighted = {}
    for filename, content in visit_contents.items():
        visit_contents_highlighted[filename] = syntax_highlight_visit_content(content)
    
    # Convert to JSON for embedding (store both raw and highlighted)
    visit_contents_raw_json = json.dumps(visit_contents)
    visit_contents_highlighted_json = json.dumps(visit_contents_highlighted)
    
    # Calculate statistics - MUST BE BEFORE HTML STRING STARTS
    total_rows = len(df)
    total_cols = len(df.columns)
    unique_visits = df['Visit_ID'].nunique() if 'Visit_ID' in df.columns else 0
    unique_sci_ids = df['SCI_ID'].nunique() if 'SCI_ID' in df.columns else 0
    
    # Calculate program/pass/segment/observation counts
    # These are hierarchical, so we need to count unique combinations
    # Use unique visits to avoid counting duplicates from multiple exposures
    if 'Visit_ID' in df.columns:
        unique_visits_df = df.drop_duplicates(subset=['Visit_ID'])
        
        # Programs: Count unique program numbers
        unique_programs = unique_visits_df['Program_Number'].nunique() if 'Program_Number' in unique_visits_df.columns else 0
        
        # Passes: Count unique (Program, Pass) combinations
        if 'Program_Number' in unique_visits_df.columns and 'Pass_Number' in unique_visits_df.columns:
            unique_passes = unique_visits_df[['Program_Number', 'Pass_Number']].drop_duplicates().shape[0]
        else:
            unique_passes = 0
        
        # Segments: Count unique (Program, Pass, Segment) combinations
        if all(col in unique_visits_df.columns for col in ['Program_Number', 'Pass_Number', 'Segment_Number']):
            unique_segments = unique_visits_df[['Program_Number', 'Pass_Number', 'Segment_Number']].drop_duplicates().shape[0]
        else:
            unique_segments = 0
        
        # Observations: Count unique (Program, Pass, Segment, Observation) combinations
        if all(col in unique_visits_df.columns for col in ['Program_Number', 'Pass_Number', 'Segment_Number', 'Observation_Number']):
            unique_observations = unique_visits_df[['Program_Number', 'Pass_Number', 'Segment_Number', 'Observation_Number']].drop_duplicates().shape[0]
        else:
            unique_observations = 0
    else:
        unique_programs = 0
        unique_passes = 0
        unique_segments = 0
        unique_observations = 0
    
    # Calculate total duration - ONLY count once per visit, not per exposure
    total_duration_seconds = 0
    total_duration_display = "N/A"
    if 'Duration' in df.columns and 'Visit_ID' in df.columns:
        try:
            # Reuse unique_visits_df if already created
            if 'unique_visits_df' not in locals():
                unique_visits_df = df.drop_duplicates(subset=['Visit_ID'])
            # Convert to numeric, coercing errors to NaN
            durations = pd.to_numeric(unique_visits_df['Duration'], errors='coerce')
            total_duration_seconds = durations.sum()
            
            # Only create display string if we have valid data
            if pd.notna(total_duration_seconds) and total_duration_seconds > 0:
                # Convert to hours, minutes, seconds
                hours = int(total_duration_seconds // 3600)
                minutes = int((total_duration_seconds % 3600) // 60)
                seconds = int(total_duration_seconds % 60)
                total_duration_display = f"{hours}h {minutes}m {seconds}s"
            else:
                total_duration_seconds = 0
                total_duration_display = "N/A"
        except Exception as e:
            print(f"Error calculating duration: {e}")
            total_duration_seconds = 0
            total_duration_display = "N/A"
    
    # Prepare sky plotter link HTML
    sky_plotter_link = ""
    if sky_plotter_html:
        sky_plotter_filename = os.path.basename(str(sky_plotter_html))
        sky_plotter_link = f'<p><strong>🌌 <a href="{sky_plotter_filename}" target="_blank" style="color: #3498db; text-decoration: none; font-weight: bold;">View Sky Plot →</a></strong></p>'
        
    # Start HTML document
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OPUP Report - {opup_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .metadata p {{
            margin: 5px 0;
            font-size: 14px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .summary-card:nth-child(2) {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        .summary-card:nth-child(3) {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}
        .summary-card:nth-child(4) {{
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        }}
        .summary-card:nth-child(5) {{
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }}
        .summary-card:nth-child(6) {{
            background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);
        }}
        .summary-card:nth-child(7) {{
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        }}
        .summary-card:nth-child(8) {{
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        }}
        .summary-card:nth-child(9) {{
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        }}
        .summary-card:nth-child(10) {{
            background: linear-gradient(135deg, #ff6e7f 0%, #bfe9ff 100%);
        }}
        .summary-label {{
            font-size: 12px;
            opacity: 0.9;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .summary-value {{
            font-size: 32px;
            font-weight: bold;
            margin: 5px 0;
        }}
        .summary-value.time {{
            font-size: 24px;
        }}
        .summary-detail {{
            font-size: 11px;
            opacity: 0.8;
            margin-top: 5px;
        }}
        .breakdown-section {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .breakdown-section h3 {{
            margin-top: 0;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
        }}
        .breakdown-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #ecf0f1;
        }}
        .breakdown-item:last-child {{
            border-bottom: none;
        }}
        .breakdown-label {{
            font-weight: 500;
            color: #34495e;
        }}
        .breakdown-value {{
            color: #7f8c8d;
            font-weight: bold;
        }}

        /* Program hierarchy styling */
        .program-header {{
            background: linear-gradient(to right, #f8f9fa 0%, #ffffff 100%);
            font-weight: 600;
            border-left: 5px solid #3498db;
            padding: 12px 15px !important;
            margin-bottom: 5px;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .program-badge {{
            display: inline-block;
            color: white;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 13px;
            font-weight: 700;
            margin-right: 10px;
            letter-spacing: 0.5px;
        }}
        .purpose-tag {{
            display: inline-block;
            background-color: #ecf0f1;
            color: #34495e;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 500;
            margin-left: 8px;
            font-style: italic;
        }}
        .purpose-tag-small {{
            display: inline-block;
            background-color: #e8f4f8;
            color: #2980b9;
            padding: 2px 6px;
            border-radius: 2px;
            font-size: 10px;
            font-weight: 500;
            margin-left: 5px;
            font-style: italic;
        }}
        .pass-item {{
            padding: 8px 15px 8px 40px !important;
            background-color: #fafafa;
            border-left: 3px solid #95a5a6;
            margin-left: 15px;
            margin-bottom: 3px;
            border-radius: 0 3px 3px 0;
        }}
        .segment-item {{
            padding: 6px 15px 6px 60px !important;
            font-size: 11px;
            color: #7f8c8d;
            font-style: italic;
            background-color: #f5f5f5;
            margin-left: 15px;
            border-radius: 0 3px 3px 0;
        }}

        .table-wrapper {{
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .table-container {{
            overflow-x: auto;
            overflow-y: auto;
            max-height: 700px;
            position: relative;
        }}
        /* Custom scrollbar styling */
        .table-container::-webkit-scrollbar {{
            height: 12px;
            width: 12px;
        }}
        .table-container::-webkit-scrollbar-track {{
            background: #f1f1f1;
            border-radius: 10px;
        }}
        .table-container::-webkit-scrollbar-thumb {{
            background: #888;
            border-radius: 10px;
        }}
        .table-container::-webkit-scrollbar-thumb:hover {{
            background: #555;
        }}
        table {{
            border-collapse: collapse;
            width: max-content;
            min-width: 100%;
            font-size: 12px;
        }}
        thead {{
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px 15px;
            text-align: left;
            white-space: nowrap;
            border-right: 1px solid #2980b9;
            font-weight: 600;
        }}
        th:last-child {{
            border-right: none;
        }}
        /* Make first few columns sticky */
        th:nth-child(1),
        td:nth-child(1) {{
            position: sticky;
            left: 0;
            z-index: 10;
            background-color: #fff;
        }}
        th:nth-child(1) {{
            background-color: #2980b9;
            z-index: 101;
        }}
        th:nth-child(2),
        td:nth-child(2) {{
            position: sticky;
            left: 0;
            z-index: 9;
            background-color: #fff;
        }}
        th:nth-child(2) {{
            background-color: #2980b9;
            z-index: 101;
        }}
        td {{
            padding: 10px 15px;
            border-bottom: 1px solid #ddd;
            border-right: 1px solid #eee;
            white-space: nowrap;
        }}
        td:last-child {{
            border-right: none;
        }}
        tr:hover td {{
            background-color: #e8f4f8;
        }}
        tr:nth-child(even) td {{
            background-color: #f9f9f9;
        }}
        tr:nth-child(even):hover td {{
            background-color: #e8f4f8;
        }}
        /* Keep sticky columns highlighted on hover */
        tr:hover td:nth-child(1),
        tr:hover td:nth-child(2) {{
            background-color: #d5e8f3;
        }}
        a.visit-link {{
            color: #2980b9;
            text-decoration: none;
            font-weight: bold;
            cursor: pointer;
        }}
        a.visit-link:hover {{
            color: #3498db;
            text-decoration: underline;
        }}
        .highlight {{
            background-color: #fff9c4;
            font-weight: 600;
        }}
        .scroll-indicator {{
            text-align: center;
            padding: 10px;
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            margin-bottom: 10px;
            font-size: 13px;
        }}
        .scroll-indicator::before {{
            content: "⟷ ";
            font-weight: bold;
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            margin-top: 10px;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-label {{
            font-size: 12px;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .footer {{
            margin-top: 30px;
            text-align: center;
            color: #7f8c8d;
            font-size: 12px;
        }}
        
        /* Modal styles */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.7);
        }}
        .modal-content {{
            background-color: #fefefe;
            margin: 1% auto;
            padding: 0;
            border: 1px solid #888;
            width: 96%;
            max-width: 1800px;
            height: 94vh;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
        }}
        .modal-header {{
            padding: 15px 20px;
            background-color: #3498db;
            color: white;
            border-radius: 8px 8px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
        }}
        .modal-header h2 {{
            margin: 0;
            font-size: 18px;
        }}
        .modal-controls {{
            display: flex;
            gap: 15px;
            align-items: center;
        }}
        .syntax-toggle {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }}
        .syntax-toggle input[type="checkbox"] {{
            width: 18px;
            height: 18px;
            cursor: pointer;
        }}
        .close {{
            color: white;
            font-size: 32px;
            font-weight: bold;
            cursor: pointer;
            line-height: 20px;
            padding: 0 10px;
        }}
        .close:hover,
        .close:focus {{
            color: #ecf0f1;
        }}
        .modal-body {{
            padding: 0;
            overflow: hidden;
            flex: 1;
            display: flex;
            flex-direction: column;
        }}
        .visit-content {{
            background-color: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
            font-family: 'Consolas', 'Courier New', Monaco, monospace;
            font-size: 11px;
            white-space: pre;
            line-height: 1.5;
            overflow: auto;
            flex: 1;
            margin: 0;
        }}
        
        /* STOL Syntax Highlighting Colors */
        .stol-keyword {{
            color: #c586c0;
            font-weight: bold;
        }}
        .stol-function {{
            color: #dcdcaa;
            font-weight: bold;
        }}
        .stol-number {{
            color: #b5cea8;
        }}
        .stol-string {{
            color: #ce9178;
        }}
        .stol-comment {{
            color: #6a9955;
            font-style: italic;
        }}
        .stol-constant {{
            color: #4fc1ff;
            font-weight: 600;
        }}
        .stol-variable {{
            color: #9cdcfe;
        }}
        .stol-param {{
            color: #f48771;
        }}
        .stol-param-name {{
            color: #9cdcfe;
        }}
        .stol-punctuation {{
            color: #d4d4d4;
        }}
        .stol-brace {{
            color: #ffd700;
            font-weight: bold;
        }}
        
        /* Custom scrollbar for visit content */
        .visit-content::-webkit-scrollbar {{
            height: 14px;
            width: 14px;
        }}
        .visit-content::-webkit-scrollbar-track {{
            background: #0d0d0d;
            border-radius: 10px;
        }}
        .visit-content::-webkit-scrollbar-thumb {{
            background: #4a4a4a;
            border-radius: 10px;
        }}
        .visit-content::-webkit-scrollbar-thumb:hover {{
            background: #5a5a5a;
        }}
        .modal-footer {{
            padding: 12px 20px;
            background-color: #ecf0f1;
            border-radius: 0 0 8px 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
        }}
        .footer-info {{
            font-size: 12px;
            color: #7f8c8d;
        }}
        .footer-buttons {{
            display: flex;
            gap: 10px;
        }}
        .download-btn {{
            background-color: #27ae60;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
        }}
        .download-btn:hover {{
            background-color: #229954;
        }}
        .copy-btn {{
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
        }}
        .copy-btn:hover {{
            background-color: #2980b9;
        }}
        .scroll-hint {{
            position: absolute;
            top: 80px;
            right: 40px;
            background-color: rgba(255, 193, 7, 0.9);
            color: #333;
            padding: 8px 15px;
            border-radius: 5px;
            font-size: 12px;
            pointer-events: none;
            animation: fadeOut 4s forwards;
        }}
        @keyframes fadeOut {{
            0% {{ opacity: 1; }}
            70% {{ opacity: 1; }}
            100% {{ opacity: 0; }}
        }}

        /* Collapsible observation group styles */
        .group-header-row {{
            cursor: pointer;
        }}
        .group-header-row td:first-child {{
            position: relative;
        }}
        .expand-btn {{
            display: inline-block;
            width: 20px;
            height: 20px;
            line-height: 20px;
            text-align: center;
            background-color: #3498db;
            color: white;
            border-radius: 3px;
            font-size: 11px;
            font-weight: bold;
            cursor: pointer;
            margin-right: 5px;
            user-select: none;
            vertical-align: middle;
        }}
        .expand-btn:hover {{
            background-color: #2980b9;
        }}
        .expand-btn.expanded {{
            background-color: #e74c3c;
        }}
        .group-child-row {{
            display: none;
            background-color: #f9f9f9 !important;
        }}
        .group-child-row.visible {{
            display: table-row;
        }}
        .group-child-row td {{
            border-top: 1px dashed #ddd !important;
        }}
        .obs-count-badge {{
            display: inline-block;
            background-color: #e67e22;
            color: white;
            padding: 1px 6px;
            border-radius: 10px;
            font-size: 10px;
            margin-left: 5px;
            vertical-align: middle;
        }}
    </style>
</head>
<body>
    <h1>OPUP Report: {opup_name}</h1>

    <div class="metadata">
        <p><strong>Archive:</strong> {opup_name}</p>
        <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
    
    # Add sky plotter link if available
    if sky_plotter_html:
        sky_plotter_filename = os.path.basename(sky_plotter_html)
        html += f"""        <p><strong>🌌 <a href="{sky_plotter_filename}" target="_blank" style="color: #3498db; text-decoration: none; font-weight: bold;">View Sky Plot →</a></strong></p>
"""
    
    html += f"""    </div>


    
    <div class="summary-grid">
        <div class="summary-card">
            <div class="summary-label">Programs</div>
            <div class="summary-value">{unique_programs}</div>
            <div class="summary-detail">Unique program numbers</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Passes</div>
            <div class="summary-value">{unique_passes}</div>
            <div class="summary-detail">Unique pass numbers</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Segments</div>
            <div class="summary-value">{unique_segments}</div>
            <div class="summary-detail">Unique segment numbers</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Observations</div>
            <div class="summary-value">{unique_observations}</div>
            <div class="summary-detail">Unique observation numbers</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Visits</div>
            <div class="summary-value">{unique_visits}</div>
            <div class="summary-detail">Distinct Visit IDs</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Total Exposures</div>
            <div class="summary-value">{total_rows}</div>
            <div class="summary-detail">All rows in data</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Total Duration</div>
            <div class="summary-value time">{total_duration_display}</div>
            <div class="summary-detail">{int(total_duration_seconds):,} seconds</div>
        </div>
    </div>
"""
# Add Program/Pass/Segment hierarchy breakdown
    if 'Visit_ID' in df.columns:
        unique_visits_df = df.drop_duplicates(subset=['Visit_ID'])
        
        # Program breakdown
        if 'Program_Number' in unique_visits_df.columns:
            html += """
    <div class="breakdown-section">
        <h3>📋 Program Hierarchy</h3>
"""
            # Define color palette for programs
            program_colors = [
                '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
                '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b'
            ]
            
            # Group by Program
            if 'Duration' in unique_visits_df.columns:
                unique_visits_df_copy = unique_visits_df.copy()
                unique_visits_df_copy['Duration'] = pd.to_numeric(unique_visits_df_copy['Duration'], errors='coerce')
                
                prog_breakdown = unique_visits_df_copy.groupby('Program_Number').agg({
                    'Visit_ID': 'count',
                    'Duration': 'sum'
                }).reset_index()
                prog_breakdown.columns = ['Program', 'Visits', 'Duration']
            else:
                prog_breakdown = unique_visits_df.groupby('Program_Number').agg({
                    'Visit_ID': 'count'
                }).reset_index()
                prog_breakdown.columns = ['Program', 'Visits']
            
            prog_breakdown = prog_breakdown.sort_values('Program')
            
            # Display each program
            for prog_idx, prog_row in enumerate(prog_breakdown.iterrows()):
                _, prog_row = prog_row
                prog_num = prog_row['Program']
                prog_visits = int(prog_row['Visits'])
                
                # Assign color to program
                prog_color = program_colors[prog_idx % len(program_colors)]
                
                # Format duration if available
                duration_str = ""
                if 'Duration' in prog_breakdown.columns and prog_row['Duration'] > 0:
                    hours = int(prog_row['Duration'] // 3600)
                    minutes = int((prog_row['Duration'] % 3600) // 60)
                    duration_str = f" | {hours}h {minutes}m"
                
                # Get intended purpose for this program if available
                prog_purpose = ""
                if 'Intended_Purpose' in unique_visits_df.columns:
                    prog_data = unique_visits_df[unique_visits_df['Program_Number'] == prog_num]
                    purposes = prog_data['Intended_Purpose'].dropna().unique()
                    if len(purposes) > 0:
                        # Take first purpose or most common if multiple
                        prog_purpose = purposes[0]
                        if len(purposes) > 1:
                            prog_purpose += f" (+{len(purposes)-1} more)"
                
                html += f"""        <div class="breakdown-item program-header" style="border-left-color: {prog_color};">
            <span class="breakdown-label">
                <span class="program-badge" style="background-color: {prog_color};">Program {prog_num}</span>
"""
                
                if prog_purpose:
                    html += f"""                <span class="purpose-tag">{prog_purpose}</span>
"""
                
                html += f"""            </span>
            <span class="breakdown-value">{prog_visits} visits{duration_str}</span>
        </div>
"""
                
                # Get passes for this program
                if 'Pass_Number' in unique_visits_df.columns:
                    prog_data = unique_visits_df[unique_visits_df['Program_Number'] == prog_num]
                    
                    if 'Duration' in prog_data.columns:
                        prog_data_copy = prog_data.copy()
                        prog_data_copy['Duration'] = pd.to_numeric(prog_data_copy['Duration'], errors='coerce')
                        
                        pass_breakdown = prog_data_copy.groupby('Pass_Number').agg({
                            'Visit_ID': 'count',
                            'Duration': 'sum'
                        }).reset_index()
                        pass_breakdown.columns = ['Pass', 'Visits', 'Duration']
                    else:
                        pass_breakdown = prog_data.groupby('Pass_Number').agg({
                            'Visit_ID': 'count'
                        }).reset_index()
                        pass_breakdown.columns = ['Pass', 'Visits']
                    
                    pass_breakdown = pass_breakdown.sort_values('Pass')
                    
                    # Display passes under this program
                    for _, pass_row in pass_breakdown.iterrows():
                        pass_num = pass_row['Pass']
                        pass_visits = int(pass_row['Visits'])
                        
                        # Format duration if available
                        pass_duration_str = ""
                        if 'Duration' in pass_breakdown.columns and pass_row['Duration'] > 0:
                            hours = int(pass_row['Duration'] // 3600)
                            minutes = int((pass_row['Duration'] % 3600) // 60)
                            pass_duration_str = f" | {hours}h {minutes}m"
                        
                        # Get intended purposes for this pass
                        pass_purposes = ""
                        if 'Intended_Purpose' in unique_visits_df.columns:
                            pass_data = prog_data[prog_data['Pass_Number'] == pass_num]
                            purposes = pass_data['Intended_Purpose'].dropna().unique()
                            if len(purposes) > 0:
                                # Show all unique purposes for this pass
                                if len(purposes) == 1:
                                    pass_purposes = f' <span class="purpose-tag-small">{purposes[0]}</span>'
                                else:
                                    pass_purposes = f' <span class="purpose-tag-small">{len(purposes)} purposes</span>'
                        
                        html += f"""        <div class="breakdown-item pass-item" style="border-left-color: {prog_color};">
            <span class="breakdown-label">└─ Pass {pass_num}{pass_purposes}</span>
            <span class="breakdown-value">{pass_visits} visits{pass_duration_str}</span>
        </div>
"""
                        
                        # Get segments for this pass (if available)
                        if 'Segment_Number' in unique_visits_df.columns:
                            pass_data = prog_data[prog_data['Pass_Number'] == pass_num]
                            segments = pass_data['Segment_Number'].unique()
                            segments = sorted([s for s in segments if pd.notna(s)])
                            
                            if len(segments) > 0:
                                if len(segments) <= 5:
                                    segment_list = ', '.join([f"S{int(s):03d}" for s in segments])
                                else:
                                    segment_list = ', '.join([f"S{int(s):03d}" for s in segments[:5]]) + f" ... (+{len(segments)-5} more)"
                                
                                html += f"""        <div class="breakdown-item segment-item">
            <span class="breakdown-label">   └─ Segments: {segment_list}</span>
            <span class="breakdown-value">{len(segments)} total</span>
        </div>
"""
            
            html += "    </div>\n"

    # Add instrument breakdown if available
    if 'Science_Instrument' in df.columns:
        html += """
    <div class="breakdown-section">
        <h3>Instrument Breakdown</h3>
"""
        # Create aggregation dictionary
        agg_dict = {
            'Visit_ID': 'nunique',
            'SCI_ID': 'count'
        }
        
        instr_breakdown = df.groupby('Science_Instrument', as_index=False).agg(agg_dict)
        instr_breakdown.columns = ['Instrument', 'Unique Visits', 'Total Exposures']
        
        # Add duration if available - only count once per visit
        if 'Duration' in df.columns and 'Visit_ID' in df.columns:
            try:
                # Get unique visits only for duration calculation
                unique_visits_df = df.drop_duplicates(subset=['Visit_ID']).copy()
                # Convert duration to numeric
                unique_visits_df['Duration'] = pd.to_numeric(unique_visits_df['Duration'], errors='coerce')
                instr_duration = unique_visits_df.groupby('Science_Instrument', as_index=False)['Duration'].sum()
                instr_duration.columns = ['Instrument', 'Total Duration']
                instr_breakdown = instr_breakdown.merge(instr_duration, on='Instrument', how='left')
                instr_breakdown['Total Duration'] = instr_breakdown['Total Duration'].fillna(0)
                
                for _, row in instr_breakdown.iterrows():
                    if row['Total Duration'] > 0:
                        hours = int(row['Total Duration'] // 3600)
                        minutes = int((row['Total Duration'] % 3600) // 60)
                        duration_str = f" | {hours}h {minutes}m"
                    else:
                        duration_str = ""
                    html += f"""        <div class="breakdown-item">
            <span class="breakdown-label">{row['Instrument']}</span>
            <span class="breakdown-value">{int(row['Unique Visits'])} visits | {int(row['Total Exposures'])} exposures{duration_str}</span>
        </div>
"""
            except Exception as e:
                print(f"Error calculating instrument duration: {e}")
                for _, row in instr_breakdown.iterrows():
                    html += f"""        <div class="breakdown-item">
            <span class="breakdown-label">{row['Instrument']}</span>
            <span class="breakdown-value">{int(row['Unique Visits'])} visits | {int(row['Total Exposures'])} exposures</span>
        </div>
"""
        else:
            # Duration column not available, display without duration
            for _, row in instr_breakdown.iterrows():
                html += f"""        <div class="breakdown-item">
            <span class="breakdown-label">{row['Instrument']}</span>
            <span class="breakdown-value">{int(row['Unique Visits'])} visits | {int(row['Total Exposures'])} exposures</span>
        </div>
"""
        html += "    </div>\n"
    
    # Add filter breakdown if available
    if 'WFI_Optical_Element' in df.columns:
        html += """
    <div class="breakdown-section">
        <h3>Filter Usage</h3>
"""
        # Create aggregation dictionary
        agg_dict = {
            'Visit_ID': 'nunique',
            'SCI_ID': 'count'
        }
        
        filter_breakdown = df.groupby('WFI_Optical_Element', as_index=False).agg(agg_dict)
        filter_breakdown.columns = ['Filter', 'Unique Visits', 'Total Exposures']
        
        # Add duration if available - only count once per visit
        if 'Duration' in df.columns and 'Visit_ID' in df.columns:
            try:
                # Get unique visits only for duration calculation
                unique_visits_df = df.drop_duplicates(subset=['Visit_ID']).copy()
                # Convert duration to numeric
                unique_visits_df['Duration'] = pd.to_numeric(unique_visits_df['Duration'], errors='coerce')
                filter_duration = unique_visits_df.groupby('WFI_Optical_Element', as_index=False)['Duration'].sum()
                filter_duration.columns = ['Filter', 'Total Duration']
                filter_breakdown = filter_breakdown.merge(filter_duration, on='Filter', how='left')
                filter_breakdown['Total Duration'] = filter_breakdown['Total Duration'].fillna(0)
                filter_breakdown = filter_breakdown.sort_values('Total Duration', ascending=False)
                
                for _, row in filter_breakdown.iterrows():
                    if pd.notna(row['Filter']):
                        if row['Total Duration'] > 0:
                            hours = int(row['Total Duration'] // 3600)
                            minutes = int((row['Total Duration'] % 3600) // 60)
                            duration_str = f" | {hours}h {minutes}m"
                        else:
                            duration_str = ""
                        html += f"""        <div class="breakdown-item">
            <span class="breakdown-label">{row['Filter']}</span>
            <span class="breakdown-value">{int(row['Unique Visits'])} visits | {int(row['Total Exposures'])} exposures{duration_str}</span>
        </div>
"""
            except Exception as e:
                print(f"Error calculating filter duration: {e}")
                filter_breakdown = filter_breakdown.sort_values('Total Exposures', ascending=False)
                for _, row in filter_breakdown.iterrows():
                    if pd.notna(row['Filter']):
                        html += f"""        <div class="breakdown-item">
            <span class="breakdown-label">{row['Filter']}</span>
            <span class="breakdown-value">{int(row['Unique Visits'])} visits | {int(row['Total Exposures'])} exposures</span>
        </div>
"""
        else:
            # Duration column not available, display without duration
            filter_breakdown = filter_breakdown.sort_values('Total Exposures', ascending=False)
            for _, row in filter_breakdown.iterrows():
                if pd.notna(row['Filter']):
                    html += f"""        <div class="breakdown-item">
            <span class="breakdown-label">{row['Filter']}</span>
            <span class="breakdown-value">{int(row['Unique Visits'])} visits | {int(row['Total Exposures'])} exposures</span>
        </div>
"""
        html += "    </div>\n"
        
    # Determine if grouping is possible (needed for both buttons and row generation)
    GROUP_COLS = ['Program_Number', 'Exec_Plan_Number', 'Pass_Number', 
                  'Segment_Number', 'Observation_Number']
    group_cols_present = all(col in df.columns for col in GROUP_COLS)
        # Add expand/collapse controls above the table
    if group_cols_present:
        html += """
    <div class="group-controls">
        <button class="expand-all-btn" onclick="toggleAllVisitGroups(true)">▼ Expand All Visits</button>
        <button class="collapse-all-btn" onclick="toggleAllVisitGroups(false)">▶ Collapse All Visits</button>
    </div>
"""

    html += f"""
    <div class="scroll-indicator">
        Scroll horizontally to view all {len(df.columns)} columns | First columns are fixed for easy reference | Click visit file names to view content
    </div>
    
    <div class="table-wrapper">
        <div class="table-container">
            <table>
                <thead>
                    <tr>
"""
    
    # Add table headers
    for col in df.columns:
        html += f"                        <th>{col}</th>\n"
    
    html += """                    </tr>
                </thead>
                <tbody>
"""
    
# ================================================================
    # Add table rows with visit-level grouping/collapsing
    # ================================================================
    # Group by Program + Exec Plan + Pass + Segment + Observation
    # Rows sharing these 5 columns are different exposures of the same visit
    # GROUP_COLS = ['Program_Number', 'Exec_Plan_Number', 'Pass_Number', 
    #               'Segment_Number', 'Observation_Number']
    
    # group_cols_present = all(col in df.columns for col in GROUP_COLS)
    
    # Helper function to format a single cell
    def _format_cell(col, value):
        if pd.isna(value):
            return ""
        elif col in ['Visit_ID', 'SCI_ID']:
            return f'<span class="highlight">{html_module.escape(str(value))}</span>'
        elif col == 'Visit_File_Name' and str(value).endswith('.vst'):
            vf = str(value)
            return (f'<a href="#" class="visit-link" onclick="showVisitContent(\'{vf}\'); '
                    f'return false;" title="Click to view {vf}">{vf}</a>')
        else:
            return html_module.escape(str(value))
    
    if group_cols_present:
        # Build group key per row and identify groups
        df_temp = df.copy()
        df_temp['_group_key'] = df_temp[GROUP_COLS].astype(str).agg('|'.join, axis=1)
        
        # Count rows per group
        group_sizes = df_temp.groupby('_group_key', sort=False).size()
        
        # Track group transitions
        current_group = None
        group_counter = 0
        row_in_group = 0
        
        for idx, row in df_temp.iterrows():
            gk = row['_group_key']
            n_rows = group_sizes[gk]
            
            if gk != current_group:
                # New group starts
                current_group = gk
                group_counter += 1
                row_in_group = 0
                group_id = f"vgrp_{group_counter}"
            else:
                row_in_group += 1
            
            # --- Determine row class ---
            if n_rows <= 1:
                # Single-exposure visit: render normally, no grouping needed
                html += "                    <tr>\n"
            elif row_in_group == 0:
                # First exposure = summary header row
                html += f'                    <tr class="group-header-row" data-group="{group_id}">\n'
            else:
                # Subsequent exposures = hidden child rows
                html += f'                    <tr class="group-child-row" data-group="{group_id}">\n'
            
            # --- Render cells ---
            for col_idx, col in enumerate(df.columns):
                value = row[col]
                cell_content = _format_cell(col, value)
                
                # Prepend expand button to the first column of multi-exposure header row
                if col_idx == 0 and n_rows > 1 and row_in_group == 0:
                    cell_content = (
                        f'<span class="expand-btn" '
                        f'onclick="toggleVisitGroup(\'{group_id}\', this); event.stopPropagation();" '
                        f'title="Click to show/hide {n_rows} exposures">▶</span>'
                        f'{cell_content}'
                        f'<span class="exp-count-badge">{n_rows} exp</span>'
                    )
                
                html += f"                        <td>{cell_content}</td>\n"
            
            html += "                    </tr>\n"
            
    else:
        # Fallback: grouping columns not available, render all rows flat
        for idx, row in df.iterrows():
            html += "                    <tr>\n"
            for col in df.columns:
                cell_content = _format_cell(col, row[col])
                html += f"                        <td>{cell_content}</td>\n"
            html += "                    </tr>\n"
    
    html += f"""                </tbody>
            </table>
        </div>
    </div>
    
    <div class="stats">
        <div class="stat-item">
            <div class="stat-label">Total Rows</div>
            <div class="stat-value">{total_rows}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Total Columns</div>
            <div class="stat-value">{total_cols}</div>
        </div>
    </div>
    
    <!-- Modal for displaying visit file content -->
    <div id="visitModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2 id="modalTitle">Visit File Content</h2>
                <div class="modal-controls">
                    <div class="syntax-toggle">
                        <label for="syntaxHighlight">Syntax Highlighting:</label>
                        <input type="checkbox" id="syntaxHighlight" checked onchange="toggleSyntaxHighlighting()">
                    </div>
                    <span class="close" onclick="closeModal()">&times;</span>
                </div>
            </div>
            <div class="modal-body">
                <div id="scrollHint" class="scroll-hint" style="display: none;">
                    ⟷ Scroll horizontally and vertically to view all content
                </div>
                <pre id="visitContentDisplay" class="visit-content"></pre>
            </div>
            <div class="modal-footer">
                <div class="footer-info">
                    <span id="fileSize"></span>
                </div>
                <div class="footer-buttons">
                    <button class="copy-btn" onclick="copyToClipboard()">Copy to Clipboard</button>
                    <button class="download-btn" onclick="downloadVisitFile()">Download</button>
                </div>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>Generated by OPUP Parser | Visit files are contained in the .tgz archive</p>
        <p>Tip: Use Shift + Mouse Wheel to scroll horizontally | Click visit file names to view content</p>
    </div>
    
    <script>
        // Store visit file contents - loaded from JSON
        const visitContentsRaw = {visit_contents_raw_json};
        const visitContentsHighlighted = {visit_contents_highlighted_json};
        
        console.log('Loaded visit contents:', Object.keys(visitContentsRaw).length, 'files');
        
        let currentVisitFile = '';
        let syntaxEnabled = true;
        
        function showVisitContent(filename) {{
            console.log('showVisitContent called with:', filename);
            
            const modal = document.getElementById('visitModal');
            const titleElement = document.getElementById('modalTitle');
            const contentElement = document.getElementById('visitContentDisplay');
            const fileSizeElement = document.getElementById('fileSize');
            const scrollHint = document.getElementById('scrollHint');
            const syntaxCheckbox = document.getElementById('syntaxHighlight');
            
            if (!modal || !titleElement || !contentElement) {{
                console.error('Modal elements not found!');
                return;
            }}
            
            currentVisitFile = filename;
            titleElement.textContent = 'Visit File: ' + filename;
            
            if (visitContentsRaw[filename]) {{
                console.log('Found content for:', filename, 'Length:', visitContentsRaw[filename].length);
                
                // Display with or without syntax highlighting
                if (syntaxEnabled && visitContentsHighlighted[filename]) {{
                    contentElement.innerHTML = visitContentsHighlighted[filename];
                }} else {{
                    contentElement.textContent = visitContentsRaw[filename];
                }}
                
                // Show file size info
                const sizeKB = (visitContentsRaw[filename].length / 1024).toFixed(2);
                const lines = visitContentsRaw[filename].split('\\n').length;
                fileSizeElement.textContent = `Size: ${{sizeKB}} KB | Lines: ${{lines}}`;
                
                // Show scroll hint briefly
                scrollHint.style.display = 'block';
                setTimeout(() => {{
                    scrollHint.style.display = 'none';
                }}, 4000);
            }} else {{
                console.warn('No content found for:', filename);
                console.log('Available files:', Object.keys(visitContentsRaw));
                contentElement.textContent = 'Visit file content not available.\\n\\nAvailable files:\\n' + 
                    Object.keys(visitContentsRaw).join('\\n');
                fileSizeElement.textContent = '';
            }}
            
            modal.style.display = 'block';
            console.log('Modal displayed');
        }}
        
        function toggleSyntaxHighlighting() {{
            const checkbox = document.getElementById('syntaxHighlight');
            syntaxEnabled = checkbox.checked;
            
            // Refresh the current display if a file is open
            if (currentVisitFile) {{
                const contentElement = document.getElementById('visitContentDisplay');
                if (syntaxEnabled && visitContentsHighlighted[currentVisitFile]) {{
                    contentElement.innerHTML = visitContentsHighlighted[currentVisitFile];
                }} else {{
                    contentElement.textContent = visitContentsRaw[currentVisitFile];
                }}
            }}
        }}
        
        function closeModal() {{
            const modal = document.getElementById('visitModal');
            modal.style.display = 'none';
            console.log('Modal closed');
        }}
        
        function downloadVisitFile() {{
            if (!currentVisitFile || !visitContentsRaw[currentVisitFile]) {{
                alert('No visit file content available to download.');
                return;
            }}
            
            const content = visitContentsRaw[currentVisitFile];
            
            // Create blob and download
            const blob = new Blob([content], {{ type: 'text/plain' }});
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = currentVisitFile;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            console.log('Downloaded:', currentVisitFile);
        }}
        
        function copyToClipboard() {{
            if (!currentVisitFile || !visitContentsRaw[currentVisitFile]) {{
                alert('No visit file content available to copy.');
                return;
            }}
            
            const content = visitContentsRaw[currentVisitFile];
            
            // Use the Clipboard API
            navigator.clipboard.writeText(content).then(() => {{
                // Temporarily change button text
                const btn = event.target;
                const originalText = btn.textContent;
                btn.textContent = 'Copied!';
                btn.style.backgroundColor = '#27ae60';
                
                setTimeout(() => {{
                    btn.textContent = originalText;
                    btn.style.backgroundColor = '#3498db';
                }}, 2000);
                
                console.log('Copied to clipboard:', currentVisitFile);
            }}).catch(err => {{
                console.error('Failed to copy:', err);
                alert('Failed to copy to clipboard');
            }});
        }}
        
        // Close modal when clicking outside of it
        window.onclick = function(event) {{
            const modal = document.getElementById('visitModal');
            if (event.target == modal) {{
                closeModal();
            }}
        }}
        
        // Close modal with Escape key
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                closeModal();
            }}
        }});
        
        // Test that everything loaded
        console.log('Script loaded successfully');

        function toggleVisitGroup(groupId, btn) {{
            var childRows = document.querySelectorAll('tr.group-child-row[data-group="' + groupId + '"]');
            var isExpanded = btn.classList.contains('expanded');
            
            for (var i = 0; i < childRows.length; i++) {{
                if (isExpanded) {{
                    childRows[i].classList.remove('visible');
                }} else {{
                    childRows[i].classList.add('visible');
                }}
            }}
            
            if (isExpanded) {{
                btn.classList.remove('expanded');
                btn.textContent = '▶';
                btn.title = btn.title.replace('hide', 'show');
            }} else {{
                btn.classList.add('expanded');
                btn.textContent = '▼';
                btn.title = btn.title.replace('show', 'hide');
            }}
        }}
        
        function toggleAllVisitGroups(expand) {{
            var buttons = document.querySelectorAll('.expand-btn');
            for (var i = 0; i < buttons.length; i++) {{
                var btn = buttons[i];
                var groupId = btn.closest('tr').getAttribute('data-group');
                var isExpanded = btn.classList.contains('expanded');
                if (expand && !isExpanded) {{
                    toggleVisitGroup(groupId, btn);
                }} else if (!expand && isExpanded) {{
                    toggleVisitGroup(groupId, btn);
                }}
            }}
        }}

    </script>
</body>
</html>
"""
    
    return html
import re

def syntax_highlight_visit_content(content):
    """
    Apply syntax highlighting to STOL visit file content.
    Highlights groups, sequences, activities, commands, parameters, etc.
    
    Args:
        content: Raw visit file content string
    
    Returns:
        HTML string with syntax highlighting
    """
    if not content or content == "Visit file content not available":
        return content
    
    lines = content.split('\n')
    highlighted_lines = []
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            highlighted_lines.append('')
            continue
        
        original_line = line
        highlighted_line = line
        
        # 1. Comment lines (starting with ;@)
        if line.strip().startswith(';@'):
            highlighted_line = f'<span class="stol-comment">{html_module.escape(line)}</span>'
        
        # 2. VISIT line
        elif line.strip().startswith('VISIT,'):
            # Parse: VISIT, V0010801001001001002, EARLY=..., LATE=..., CUTOFF=..., CONVST=...;
            match = re.match(r'^(VISIT,\s*)([^,]+)(,\s*.+)$', line)
            if match:
                keyword, visit_id, rest = match.groups()
                # Parse the rest for key=value pairs
                rest_parts = []
                current = ""
                for char in rest:
                    if char == ',' or char == ';':
                        if '=' in current:
                            key, val = current.split('=', 1)
                            rest_parts.append(f'<span class="stol-param-name">{html_module.escape(key)}</span>=<span class="stol-string">{html_module.escape(val)}</span>')
                        else:
                            rest_parts.append(html_module.escape(current))
                        rest_parts.append(char)
                        current = ""
                    else:
                        current += char
                if current:
                    if '=' in current:
                        key, val = current.split('=', 1)
                        rest_parts.append(f'<span class="stol-param-name">{html_module.escape(key)}</span>=<span class="stol-string">{html_module.escape(val)}</span>')
                    else:
                        rest_parts.append(html_module.escape(current))
                
                rest_highlighted = ''.join(rest_parts)
                highlighted_line = f'<span class="stol-keyword">{html_module.escape(keyword)}</span><span class="stol-constant">{html_module.escape(visit_id)}</span>{rest_highlighted}'
            else:
                highlighted_line = f'<span class="stol-keyword">{html_module.escape(line)}</span>'
        
        # 3. GROUP line
        elif re.match(r'^\s*GROUP,', line):
            # Parse: GROUP, 01, CONGRP=NONE;
            match = re.match(r'^(\s*)(GROUP)(,\s*)(\d+)(,\s*)([A-Z_]+)(=)([^;]+)(;?)(.*)$', line)
            if match:
                indent, keyword, comma1, number, comma2, param_name, equals, param_value, semicolon, comment = match.groups()
                highlighted_line = (f'{indent}<span class="stol-keyword">{keyword}</span>{comma1}'
                                  f'<span class="stol-number">{number}</span>{comma2}'
                                  f'<span class="stol-param-name">{param_name}</span>{equals}'
                                  f'<span class="stol-constant">{param_value}</span>'
                                  f'<span class="stol-punctuation">{semicolon}</span>{comment}')
            else:
                highlighted_line = f'<span class="stol-keyword">{html_module.escape(line)}</span>'
        
        # 4. SEQ line
        elif re.match(r'^\s*SEQ,', line):
            # Parse: SEQ, 1, CONSEQ=NONE;
            match = re.match(r'^(\s*)(SEQ)(,\s*)(\d+)(,\s*)([A-Z_]+)(=)([^;]+)(;?)(.*)$', line)
            if match:
                indent, keyword, comma1, number, comma2, param_name, equals, param_value, semicolon, comment = match.groups()
                highlighted_line = (f'{indent}<span class="stol-keyword">{keyword}</span>{comma1}'
                                  f'<span class="stol-number">{number}</span>{comma2}'
                                  f'<span class="stol-param-name">{param_name}</span>{equals}'
                                  f'<span class="stol-constant">{param_value}</span>'
                                  f'<span class="stol-punctuation">{semicolon}</span>{comment}')
            else:
                highlighted_line = f'<span class="stol-keyword">{html_module.escape(line)}</span>'
        
        # 5. ACT line with function call
        elif re.match(r'^\s*ACT,', line):
            # Parse: ACT, 01, FUNCTION_NAME(...); PARAM1,PARAM2,...
            match = re.match(r'^(\s*)(ACT)(,\s*)(\d+)(,\s*)([A-Z_]+)(\([^)]*\))(;?)(.*)$', line)
            if match:
                indent, keyword, comma1, number, comma2, func_name, params_with_parens, semicolon, comment = match.groups()
                
                # Extract parameters from parentheses
                params_match = re.match(r'\(([^)]*)\)', params_with_parens)
                if params_match:
                    params_str = params_match.group(1)
                    
                    # Highlight individual parameters
                    if params_str.strip():
                        param_parts = []
                        # Split by comma, but be careful with quoted strings
                        current_param = ""
                        in_quotes = False
                        for char in params_str:
                            if char == '"' and (not current_param or current_param[-1] != '\\'):
                                in_quotes = not in_quotes
                                current_param += char
                            elif char == ',' and not in_quotes:
                                param_parts.append(current_param.strip())
                                current_param = ""
                            else:
                                current_param += char
                        if current_param:
                            param_parts.append(current_param.strip())
                        
                        # Highlight each parameter
                        highlighted_params = []
                        for param in param_parts:
                            param = param.strip()
                            # Check if it's a number
                            if re.match(r'^-?\d+\.?\d*([eE][+-]?\d+)?$', param):
                                highlighted_params.append(f'<span class="stol-number">{html_module.escape(param)}</span>')
                            # Check if it's a quoted string
                            elif param.startswith('"') and param.endswith('"'):
                                highlighted_params.append(f'<span class="stol-string">{html_module.escape(param)}</span>')
                            # Check if it's a constant/keyword
                            elif param.isupper() or param in ['TRUE', 'FALSE']:
                                highlighted_params.append(f'<span class="stol-constant">{html_module.escape(param)}</span>')
                            else:
                                highlighted_params.append(f'<span class="stol-param">{html_module.escape(param)}</span>')
                        
                        params_highlighted = '(' + ','.join(highlighted_params) + ')'
                    else:
                        params_highlighted = '()'
                else:
                    params_highlighted = html_module.escape(params_with_parens)
                
                # Highlight comment (parameter names)
                if comment.strip():
                    comment_highlighted = f'<span class="stol-comment">{html_module.escape(comment)}</span>'
                else:
                    comment_highlighted = comment
                
                highlighted_line = (f'{indent}<span class="stol-keyword">{keyword}</span>{comma1}'
                                  f'<span class="stol-number">{number}</span>{comma2}'
                                  f'<span class="stol-function">{func_name}</span>{params_highlighted}'
                                  f'<span class="stol-punctuation">{semicolon}</span>{comment_highlighted}')
            else:
                # Try simpler ACT format without parameters
                match2 = re.match(r'^(\s*)(ACT)(,\s*)(\d+)(,\s*)([A-Z_]+)(;?)(.*)$', line)
                if match2:
                    indent, keyword, comma1, number, comma2, func_name, semicolon, comment = match2.groups()
                    if comment.strip():
                        comment_highlighted = f'<span class="stol-comment">{html_module.escape(comment)}</span>'
                    else:
                        comment_highlighted = comment
                    highlighted_line = (f'{indent}<span class="stol-keyword">{keyword}</span>{comma1}'
                                      f'<span class="stol-number">{number}</span>{comma2}'
                                      f'<span class="stol-function">{func_name}</span>'
                                      f'<span class="stol-punctuation">{semicolon}</span>{comment_highlighted}')
                else:
                    highlighted_line = html_module.escape(line)
        
        # 6. Standalone comment/empty comment line (just semicolon)
        elif line.strip() == ';':
            highlighted_line = f'<span class="stol-comment">{line}</span>'
        
        else:
            # Leave other lines as-is (escape HTML)
            highlighted_line = html_module.escape(line)
        
        highlighted_lines.append(highlighted_line)
    
    return '\n'.join(highlighted_lines)
def write_to_HTML(df, output_html, opup_filepath, keep_GW=True):
    """
    Write DataFrame to HTML file with optional GW column removal.
    
    Args:
        df: DataFrame to export
        output_html: Path to output HTML file
        opup_filepath: Path to OPUP archive
        keep_GW: Whether to keep Guide Window columns
    """
    if not keep_GW:
        gw_cols = get_current_gw_columns(df)
        df = df.drop(columns=gw_cols)
    
    html_content = generate_html_report(df, opup_filepath)
    
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)


# Modified process functions to add HTML export option
def process_OPUPs_html(opup_filepaths, output_dir=None, keep_GW=True):
    """
    Process OPUP files and generate HTML reports.
    """
    for opup_filepath in opup_filepaths:
        if output_dir is None:
            output_dir = Path(opup_filepath).parent.as_posix()
            if not Path(output_dir).is_dir():
                output_dir = find_nontgz_parent(output_dir)
        
        output_html = Path(output_dir).joinpath(Path(opup_filepath).stem + '_report.html').as_posix()
        
        # Parse the opup
        opup_info = parse_OPUP(opup_filepath)
        
        # Move priority columns to the left
        opup_info = prioritize_columns(opup_info, PRIORITY_COLUMNS)
        
        # Write to HTML
        if len(opup_info.columns) > 0:
            write_to_HTML(opup_info, output_html, opup_filepath, keep_GW=keep_GW)
        else:
            print(f'No columns were returned for {opup_filepath}')


def export_unique_visits_for_plotter(df, output_csv):
    """
    Export one row per unique visit for use with roman_plotter.py
    Includes only the columns needed for sky plotting: RA, DEC, Visit_ID, etc.
    
    Args:
        df: Full DataFrame with all exposures
        output_csv: Path to output CSV file (string or Path object)
    """
    # Convert Path to string if necessary
    output_csv = str(output_csv)
    
    if 'Visit_ID' not in df.columns:
        print("Warning: No Visit_ID column found, cannot create unique visits CSV")
        return
    
    # Get unique visits only
    unique_visits_df = df.drop_duplicates(subset=['Visit_ID']).copy()
    
    # Select columns relevant for sky plotting
    columns_to_keep = []
    possible_columns = [
        'Visit_ID', 'RA', 'DEC', 'Position_Angle', 
        'Program_Number', 'Pass_Number', 'Segment_Number', 
        'Observation_Number', 'Visit_Number',
        'Start', 'Duration', 'Science_Instrument', 
        'WFI_Optical_Element', 'Visit_File_Name',
        'Earliest_Start_Time', 'Latest_Start_Time',
        'Intended_Purpose'  # Add this for color-coding in the plotter
    ]
    
    for col in possible_columns:
        if col in unique_visits_df.columns:
            columns_to_keep.append(col)
    
    # Create the filtered dataframe
    plotter_df = unique_visits_df[columns_to_keep].copy()
    
    # Save to CSV
    plotter_df.to_csv(output_csv, index=False)
    print(f"  📊 Generated sky plotter CSV: {output_csv} ({len(plotter_df)} unique visits)")
    
    return output_csv

def generate_integrated_report(opup_filepath, output_dir=None, keep_GW=True):
    """
    Generate both the detailed OPUP HTML report and the sky plotter visualization.
    
    Args:
        opup_filepath: Path to OPUP .tgz archive
        output_dir: Output directory (defaults to same as OPUP)
        keep_GW: Whether to keep Guide Window columns
    
    Returns:
        Tuple of (html_report_path, sky_plotter_path, csv_path)
    """
    from pathlib import Path
    from datetime import datetime, timezone
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path(opup_filepath).parent
        if not output_dir.is_dir():
            output_dir = find_nontgz_parent(opup_filepath)
    
    output_dir = Path(output_dir)
    opup_stem = Path(opup_filepath).stem.replace('.tgz', '').replace('.tar', '')
    
    print(f"\n{'='*60}")
    print(f"Generating Integrated OPUP Report")
    print(f"{'='*60}\n")
    
    # 1. Parse OPUP
    print("Step 1: Parsing OPUP...")
    opup_info = parse_OPUP(opup_filepath)
    opup_info = prioritize_columns(opup_info, PRIORITY_COLUMNS)
    
    # 2. Extract date from first visit for Sun position calculation
    sun_date = None
    if 'Start' in opup_info.columns:
        # Get the first non-null Start time
        first_start = opup_info['Start'].dropna().iloc[0] if len(opup_info['Start'].dropna()) > 0 else None
        
        if first_start:
            # Parse the date - handle formats like "2026-276-13:00:51 TAI"
            import re
            # YYYY-DDD format (day of year)
            doy_match = re.match(r'^(\d{4})-(\d{1,3})(?:-(\d{2}:\d{2}:\d{2}))?\s*(?:TAI|UTC|TDB|TT)?$', str(first_start).strip(), re.I)
            if doy_match:
                year = int(doy_match.group(1))
                doy = int(doy_match.group(2))
                if 1 <= doy <= 366:
                    from datetime import timedelta
                    sun_date = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
                    print(f"  ☀️  Using date from first visit: {sun_date.strftime('%Y-%m-%d')} (DOY {doy})")
            else:
                # Try other common formats
                for fmt in ('%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'):
                    try:
                        sun_date = datetime.strptime(str(first_start).split()[0], fmt).replace(tzinfo=timezone.utc)
                        print(f"  ☀️  Using date from first visit: {sun_date.strftime('%Y-%m-%d')}")
                        break
                    except ValueError:
                        continue
    
    if sun_date is None:
        sun_date = datetime.now(timezone.utc)
        print(f"  ⚠️  Could not parse visit date, using today: {sun_date.strftime('%Y-%m-%d')}")
    
    # 3. Generate unique visits CSV for sky plotter
    print("\nStep 2: Creating sky plotter CSV...")
    plotter_csv = output_dir / f"{opup_stem}_skyplot.csv"
    export_unique_visits_for_plotter(opup_info, plotter_csv)
    
    # 4. Generate sky plotter HTML by importing roman_plotter
    print("\nStep 3: Generating sky plotter...")
    sky_plotter_html = output_dir / f"{opup_stem}_skyplot.html"
    
    # Find roman_plotter.py in the same directory as this script
    script_dir = Path(__file__).parent
    roman_plotter_path = script_dir / "roman_plotter.py"
    
    if roman_plotter_path.exists():
        try:
            # Import roman_plotter functions directly
            import sys
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))
            import roman_plotter
            
            # Call the generate function directly
            # First, load the CSV data
            import pandas as pd
            plotter_data = pd.read_csv(plotter_csv)
            
            # Create a dataset in the format roman_plotter expects
            import json
            data_json = plotter_data.to_json(orient='records')
            preloaded_datasets = [{
                'fileName': plotter_csv.name,
                'data_json': data_json
            }]
            
            # Calculate Sun position for the visit date
            sun_position = roman_plotter.get_sun_position(sun_date)
            print(f"  ☀️  Sun RA={sun_position['ra']:.2f}°, Dec={sun_position['dec']:.2f}° "
                  f"(Galactic: l={sun_position['l']:.2f}°, b={sun_position['b']:.2f}°)")
            
            # Generate the HTML
            html_content = roman_plotter.generate_html(
                preloaded_datasets=preloaded_datasets,
                sun_position=sun_position
            )
            
            # Write to file
            with open(sky_plotter_html, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"  ✅ Generated sky plotter: {sky_plotter_html}")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not generate sky plotter: {e}")
            import traceback
            traceback.print_exc()
            sky_plotter_html = None
    else:
        print(f"  ⚠️  Warning: roman_plotter.py not found at {roman_plotter_path}")
        sky_plotter_html = None
    
    # 5. Generate detailed HTML report with link to sky plotter
    print("\nStep 4: Generating detailed HTML report...")
    html_report = output_dir / f"{opup_stem}_report.html"
    html_content = generate_html_report(opup_info, opup_filepath, sky_plotter_html)
    
    with open(html_report, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # 6. Optionally generate full CSV
    print("\nStep 5: Generating full CSV...")
    full_csv = output_dir / f"{opup_stem}_full.csv"
    write_to_CSV(opup_info, full_csv, keep_GW=keep_GW)
    
    print(f"\n{'='*60}")
    print(f"✅ Complete!")
    print(f"{'='*60}")
    print(f"\n📄 Detailed Report: {html_report}")
    if sky_plotter_html:
        print(f"🌌 Sky Plot:        {sky_plotter_html}")
    print(f"📊 CSV Data:        {full_csv}")
    print(f"📊 Sky Plot CSV:    {plotter_csv}")
    print(f"\nOpen {html_report} in your browser to get started!\n")
    
    return html_report, sky_plotter_html, full_csv

# Add HTML option to command-line parser
def setup_parser_with_html():
    parser = argparse.ArgumentParser(description='Parse OPUP files.')
    parser.add_argument('-opup', '--opup_filepath', type=str, nargs='+', help='Path(s) to the OPUP file(s)', default=[])
    parser.add_argument('-scf', '--scf_filepath', type=str, nargs='+', help='Path(s) to the SCF file(s)', default=[])
    parser.add_argument('-visit', '--visit_filepath', type=str, nargs='+', help='Path(s) to the visit file(s)', default=[])
    parser.add_argument('-odir', '--output_dir', type=str, help='Output file directory', default=None)
    parser.add_argument('--keep_GW', action='store_true', help='Keep Guide Window information in the output.')
    parser.add_argument('--format', type=str, choices=['csv', 'html', 'both'], default='html', 
                       help='Output format: csv, html, or both')
    return parser


def setup_parser():
    parser = argparse.ArgumentParser(description='Parse OPUP files.')
    parser.add_argument('-opup', '--opup_filepath', type=str, nargs='+', help='Path(s) to the OPUP file(s)', default=[])
    parser.add_argument('-scf', '--scf_filepath', type=str, nargs='+', help='Path(s) to the SCF file(s)', default=[])
    parser.add_argument('-visit', '--visit_filepath', type=str, nargs='+', help='Path(s) to the visit file(s)', default=[])
    parser.add_argument('-odir', '--output_dir', type=str, help='Output file directory', default=None)
    parser.add_argument('--keep_GW', action='store_true', help='Keep Guide Window information in the output.')
    parser.add_argument('--format', type=str, choices=['csv', 'html', 'both', 'integrated'], default='integrated', 
                       help='Output format: csv, html, both, or integrated (html + sky plot)')
    return parser


if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()
    
    # Extract args
    opup_filepaths = args.opup_filepath
    scf_filepaths = args.scf_filepath
    visit_filepaths = args.visit_filepath
    output_dir = args.output_dir
    keep_GW = args.keep_GW
    output_format = args.format
    
    # Process based on format
    if output_format == 'integrated':
        # Generate integrated report with sky plotter
        for opup_filepath in opup_filepaths:
            generate_integrated_report(opup_filepath, output_dir, keep_GW)
    else:
        # Original workflow
        if output_format in ['csv', 'both']:
            process_OPUPs(opup_filepaths, output_dir, keep_GW)
            process_SCFs(scf_filepaths, output_dir, keep_GW)
            process_visits(visit_filepaths, output_dir, keep_GW)
        
        if output_format in ['html', 'both']:
            process_OPUPs_html(opup_filepaths, output_dir, keep_GW)


# %%
