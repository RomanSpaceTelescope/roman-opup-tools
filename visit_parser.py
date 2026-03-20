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

def find_nontgz_parent(folderpath):
    # Returning the first path in the parents of the given path
    # that is a directory.
    for path in Path(folderpath).parents:
        if Path(path).is_dir():
            return path

def write_to_CSV(df, output_csv, keep_GW=False):

    # Extract the guide window data into a separate data frame
    if not keep_GW:
        # Splitting science and guide window columns
        sci_df, gw_df = split_df_columns(df, get_current_gw_columns(df))

        # Adding some basic identifying info to the GW df
        shared_cols = ['SCI_ID', 'GWID']
        gw_df = pd.concat((sci_df[shared_cols], gw_df), axis=1)
        # for shared_col in shared_cols:
        #     gw_df.insert(0, shared_col, sci_df[shared_col])

        # Writing to CSVs
        sci_df.to_csv(output_csv, index=False)
        gw_df.to_csv(output_csv.replace('.csv', '_GWInfo.csv'), index=False)

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

    for opup_filepath in opup_filepaths:
        if output_dir is None:
            output_dir = Path(opup_filepath).parent.as_posix()
        output_csv = Path(output_dir).joinpath(Path(opup_filepath).stem + '_csv.csv').as_posix()

        write_to_CSV(parse_OPUP(opup_filepath), output_csv, keep_GW=keep_GW)

def process_SCFs(scf_filepaths, output_dir=None, keep_GW=True):

    for scf_filepath in scf_filepaths:
        if output_dir is None:
            output_dir = Path(scf_filepath).parent.as_posix()

            # If the SCF file is stored in a gzipped archive, we have to modify the save path to accommodate this.
            if not Path(output_dir).is_dir():
                output_dir = find_nontgz_parent(output_dir)

        output_csv = Path(output_dir).joinpath(Path(scf_filepath).stem + '_csv.csv').as_posix()

        # Parsing SCF
        scf_info = parse_SCF(scf_filepath)

        # Writing to CSV
        if len(scf_info.columns)>0:
            write_to_CSV(scf_info, output_csv, keep_GW=keep_GW)

def process_visits(visit_filepaths, output_dir=None, keep_GW=True):

    for visit_filepath in visit_filepaths:
        if output_dir is None:
            output_dir = Path(visit_filepath).parent.as_posix()

            # If the visit file is stored in a gzipped archive, we have to modify the save path to accommodate this.
            if not Path(output_dir).is_dir():
                output_dir = find_nontgz_parent(output_dir)

        output_csv = Path(output_dir).joinpath(Path(visit_filepath).stem + '_csv.csv').as_posix()

        visit_info = parse_visit_file(visit_filepath)

        if len(visit_info.columns) > 0:
            write_to_CSV(visit_info, output_csv, keep_GW=keep_GW)

#%%


if __name__=='__main__':

    # Parsing cmd args
    parser = setup_parser()
    args = parser.parse_args()

    # Extracting args
    opup_filepaths = args.opup_filepath
    scf_filepaths = args.scf_filepath
    visit_filepaths = args.visit_filepath
    output_dir = args.output_dir
    keep_GW = args.keep_GW
    
    # Processing OPUP filepaths
    process_OPUPs(opup_filepaths, output_dir, keep_GW)

    # Processing SCF filepaths
    process_SCFs(scf_filepaths, output_dir, keep_GW)

    # Procesing visit filepaths
    process_visits(visit_filepaths, output_dir, keep_GW)


# %%
