import re
import YZS_lib
import sys
from multiprocessing import Pool
from multiprocessing import cpu_count
from scipy.optimize import minimize
import numpy as np


if __name__ == "__main__":
    file_path = r"D:\Calculation_Results\(Brecher C et al, 1967)\template_sym.txt"
    with open (file_path, 'r') as file:
        lines = file.readlines()
        
    processed_lines = []
    for line in lines:
        #Remove the commend from the line
        line_without_comment = re.sub(r'#.*', '', line)
        
        #If a commented line still has content, append it.
        if line_without_comment.strip() and re.search(r'#.*', line):
            processed_lines.append(line_without_comment)
        
        #Append lines that did not contain comment.
        elif not re.search(r'#.*', line):
            processed_lines.append(line_without_comment)           
    #Replace the original list with the new, cleaned list
    lines = processed_lines
    #This logic is necessary because if a line contains only a comment(text following a '#'),
    #removing it will create an empty line. These empty lines would break the
    #data blocks for energies or crystal field parameters."
        
    lines = [line.upper() for line in lines]     #Capitalise strings
    #No further modifications to lines.  

    # --- BEGIN: Input File Syntax Validation ---
    
    # This section validates the following:
    # - The format of the option section is correct.
    # - The number of options is correct.
    # - The format of the energy section is correct.

    #asterisk
    ast_num = 0
    for line in lines:
        if re.fullmatch(r"\*+", line.strip()):
            ast_num += 1
        if ast_num == 2:
            print("ERROR")
            print("Found more than one line of asterisks.")
            raise SyntaxError
    if ast_num == 0:
        print("ERROR")
        print("No section break found. Consider adding a line of asterisks between options and energies.")
        raise SyntaxError
    
    # Our file has two types of data blocks: crystal field parameter blocks and energy blocks.
    # The crystal field parameter blocks allow a user to directly specify the parameters,
    # but this option cannot be used when a symmetry is also defined. Every line
    # in this block requires a four-space indentation, and the block is terminated by any
    # line that doesn't match the pattern, which allows other content to follow immediately.
    # On the other hand, the energy blocks, which correspond to different term symbols,
    # strictly require that all lines have no indentation, and they end as soon as a
    # blank line appears.

    ALT_num = 0
    ION_num = 0
    SYM_num = 0
    PAR_num = 0
    PAR_exist = False
    BOU_num = 0
    TIME_num = 0
     
    #specification  
    in_block = False    
    for line in lines:
        if in_block:
            if re.fullmatch(r"\s{4}(AR\d-?\d)", line.rstrip()):
                PAR_exist = True
                continue
            elif re.fullmatch(r"\s{4}(AR\d-?\d)\s+\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", line.rstrip()):
                PAR_exist = True
                continue
            # elif line.isspace():
            #     PAR_num += 1
            #     in_block = False
            #     continue
            # else:
            #     print("Invalid line detected:")
            #     print(line)
            #     raise SyntaxError
            
            #Other content can immediately follow the parameters block without a blank line.
            else:
                if not PAR_exist:
                    print("Error: The PARAMETER block cannot be empty.")
                    raise SyntaxError
                PAR_num += 1
                in_block = False
            
        if re.fullmatch(r"\*+", line.rstrip()):
            break
        elif re.fullmatch(r"ION\s+(.+)", line.rstrip()):
            ION_num += 1
        elif re.fullmatch(r"SYMMETRY\s+(.+)", line.rstrip()):
            SYM_num += 1
        elif re.fullmatch(r"PARAMETERS", line.rstrip()):
            in_block = True
        elif re.fullmatch(r"BOUNDS\s+\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", line.rstrip()):
        #r"BOUNDS\s+\(\s*(-?\d+\.\d+|-?\d+)\s*,\s*(-?\d+\.\d+|-?\d+)\s*\)"
            BOU_num += 1 
        elif re.fullmatch(r"MCTIME\s+(\d+(?:\.\d+)?)", line.rstrip()):
            TIME_num += 1
        elif re.fullmatch(r"ALT_THET\s+(TRUE|FALSE)", line.rstrip()):
            ALT_num += 1
        elif line.isspace():
            continue
        else:
            print("Invalid line detected:")
            print(line)
            raise SyntaxError
            
    if ION_num == 0:
        print('No ION found.')
        raise SyntaxError
    elif SYM_num == 0 and PAR_num == 0:
        print("No Symmetry specified.")
        raise SyntaxError
    elif SYM_num == 1 and PAR_num == 1:
        print("Don't specify symmetry and allowed parameters together.")
        raise SyntaxError
    elif ION_num > 1:
        print('More than one ION found.')
        raise SyntaxError
    elif SYM_num > 1:
        print('More than one SYMMETRY found.')
        raise SyntaxError
    elif PAR_num > 1:
        print("More than one PARAMETER found.")
        raise SyntaxError
    elif BOU_num > 1:
        print('More than one BOUNDS found.')
        raise SyntaxError
    elif TIME_num > 1:
        print("More than one TIME found.")
        raise SyntaxError
        
    #energies
    in_block = False
    start = False
    has_energy = False
    for line in lines:
        if re.fullmatch(r"\*+", line.strip()):
            start = True
            continue
        
        if start == True:
            if in_block == True:
                if re.fullmatch(r"-?\d+(\.\d+)?", line.rstrip()):  
                    has_energy = True
                    continue
                elif re.fullmatch(r"NONE", line.rstrip()):
                    continue
                #A blank line is required to separate different term symbol blocks.
                elif line.isspace():
                    in_block = False
                    continue
                else:
                    print("Invalid line detected:")
                    print(line)
                    raise SyntaxError
                    
            
            
            if line.isspace():
                continue
            elif re.fullmatch(r'\d+[A-Z](\d+\/\d+|\d+)', line.rstrip()):
                in_block = True
                continue
            else:
                print("Invalid line detected:")
                print(line)
                raise SyntaxError
                
    if not has_energy:
        print("No energy found.")
        raise SyntaxError
    
    # --- END: Input File Syntax Validation ---
        

    # --- START: Extract Information ---
    
    # Additional validations:
    # - lower bound < higher bound
    # - Supported SYMMETRY
    # - Non-zero TIME
    # - Default TIME if not specified
    # - Valid ION type
    # - Alternative theta exists
    
    #specification
    in_block = False 
    Ar_dict = {}
    individual_bounds = []
    shared_bounds = None
    symmetry = None
    time = None
    alt_thet_enabled = None
    for line in lines:
        if in_block:
            # print(line)
            if m := re.fullmatch(r"\s{4}(AR\d-?\d)", line.rstrip()):
                if m.group(1) in Ar_dict:
                    print(f"Duplicate crystal field parameters: {m.group(1)}")
                    raise ValueError
                Ar_dict[m.group(1)] = None
                individual_bounds.append(None)
                continue
            elif m := re.fullmatch(r"\s{4}(AR\d-?\d)\s+\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", line.rstrip()):
                if m.group(1) in Ar_dict:
                    print(f"Duplicate crystal field parameters: {m.group(1)}")
                    raise ValueError
                Ar_dict[m.group(1)] = None
                individual_bounds.append((float(m.group(2)), float(m.group(3))))
                continue
            # elif line.isspace():
            #     in_block = False
            #     if len(Ar_dict) == 0:
            #         print("No allowed crystal field parameters found.")
            #         raise SyntaxError
            #     continue
            # else:
            #     print("Invalid line detected:")
            #     print(line)
            #     raise SyntaxError
            
            #Content may immediately follow the parameters block.
            else:
                in_block = False
                if len(Ar_dict) == 0:
                    print('No crystal field parameters found in the file')
                    raise SyntaxError
            continue
    
            
        if m := re.fullmatch(r"\*+", line.rstrip()):
            break
        elif m := re.fullmatch(r"ION\s+(.+)", line.rstrip()):
            ion = m.group(1)
        elif m := re.fullmatch(r"SYMMETRY\s+(.+)", line.rstrip()):
            symmetry = m.group(1)
        elif m := re.fullmatch(r"PARAMETERS", line.rstrip()):
            in_block = True
        elif m := re.fullmatch(r"BOUNDS\s+\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", line.rstrip()):
        #r"BOUNDS\s+\(\s*(-?\d+\.\d+|-?\d+)\s*,\s*(-?\d+\.\d+|-?\d+)\s*\)"
            shared_bounds = (float(m.group(1)), float(m.group(2))) 
        elif m := re.fullmatch(r"MCTIME\s+(\d+(?:\.\d+)?)", line.rstrip()):
            time = float(m.group(1))
        elif m := re.fullmatch(r"ALT_THET\s+(TRUE|FALSE)", line.rstrip()):
            alt_thet_enabled = m.group(1) == "TRUE"
        elif line.isspace():
            continue
        # else:
        #     print("Invalid line detected:")
        #     print(line)
        #     raise SyntaxError
        

        

    if type(shared_bounds) == tuple and shared_bounds[0] >= shared_bounds[1]:
        raise ValueError(f"Invalid bounds: {shared_bounds}")
        
    if type(symmetry) == str and symmetry not in [item.upper() for item in YZS_lib.allowed_symmetries]:
        print(f'Symmetry not allowed: {symmetry}')
        raise ValueError
        
    if time == 0:
        print("Execution time cannot be zero.")
        raise ValueError
    elif time == None:
        time = 15
    
    if ion not in {key[0].upper() for key in YZS_lib.Thet.keys()}:
        print(f'Ion not allowed: {ion}')
        raise ValueError
        
    # if symmetry in YZS_lib.cubic_symmetry and ion == 'TB3+' and any(key != '7F6' for key in EXNRG):
    #     print("Cubic symmetry isn't supported for Tb3+")
    #     sys.exit()
    # if ion == 'TB3+' and Ar_dict:
    #     for item in Ar_dict:
    #         if item[2] == '2':
    #             print("Second order crystal field prameters aren't supported for   ")
    #             sys.exit()
                
    assert len(Ar_dict) == len(individual_bounds)
    
    if alt_thet_enabled:
        if ion not in {key[0].upper() for key in YZS_lib.alt_Thet.keys()}:
            print(f"No alternative theta values for {ion}.")
            print("Ions with alternative theta values ininclude:")
            print({key[0] for key in YZS_lib.alt_Thet.keys()})
            raise ValueError

        

    #energy
    in_block = False
    block_name = None
    start = False
    EXNRG = {}
    #has_energy = False
    for line in lines:
        if re.fullmatch(r"\*+", line.strip()):
            start = True
            continue
        
        if start == True:
            if in_block == True:
                if re.fullmatch(r"-?\d+(\.\d+)?", line.rstrip()):  
                    # has_energy = True
                    EXNRG[block_name].append(float(line.rstrip()))
                    continue
                elif re.fullmatch(r"NONE", line.rstrip()):
                    EXNRG[block_name].append(line.rstrip())
                    continue
                #A blank line is required to separate different term symbol blocks.
                elif line.isspace():
                    if len(EXNRG[block_name]) == 0:
                        print('No energy found for ' + block_name)
                        raise SyntaxError
                    in_block = False
                    continue
                # else:
                #     print("Invalid line detected:")
                #     print(line)
                #     raise SyntaxError
                    
            
            
            if line.isspace():
                continue
            elif re.fullmatch(r'\d+[A-Z](\d+\/\d+|\d+)', line.rstrip()):
                #print(line)
                in_block = True
                if line.rstrip() in EXNRG:
                    print("Duplicate term symbol found.")
                    print(line.rstrip())
                    raise SyntaxError
                else:
                    EXNRG[line.rstrip()] = []
                    block_name = line.rstrip()
                continue
            # else:
            #     print("Invalid line detected:")
            #     print(line)
            #     raise SyntaxError
            
    # --- END: Extract Information ---
    # Information will be further checked and processed for fitting.
            
    for key, value in EXNRG.items():
        m = re.match(r'\d+[A-Z](\d+\/\d+|\d+)', key)
        J = m.group(1)
        if '/' in J:
            numerator, denominator = J.split('/')
            J = int(numerator) / int(denominator)
        else:
            J = int(J)
        if type(symmetry) == str and len(value) != YZS_lib.get_stark_levels_count(site_symmetry = symmetry, J = J):
            print("Dubious degeneracy for term " + key + '.')
            raise ValueError
        
    if symmetry not in YZS_lib.cubic_symmetry and ion == 'TB3+' and any(key != '7F6' for key in EXNRG):
        print("Cubic symmetry isn't supported for excited states of Tb3+.")
        sys.exit()
    # ground state '7F6' is provided by (Stevens, 1952) and (Hutchings, 1964)
    # This is the best the author can offer, as I don't know how to calculate the theta value.
    if ion == 'TB3+' and Ar_dict and any(key != '7F6' for key in EXNRG):
        for item in Ar_dict:
            if item[2] == '2':
                print("Second order crystal field parameters aren't supported for excited states of Tb3+")
                sys.exit()    
    
    # Energy lists with missing energies will not be sorted. The user should provide an already-sorted energy list themselves.
    # It is possible to sort an energy list when all energies are provided. However, it is highly recommended that the user do it themselves.
    for key in EXNRG:
        if all(isinstance(item, float) for item in EXNRG[key]):
            if EXNRG[key] != sorted(EXNRG[key]):
                print(f"Warning: The energy list for term '{key}' was not sorted.")
                EXNRG[key].sort()
        
    for key, value in EXNRG.items():
        # normalises the energies with respect to the first energy.
        if all(isinstance(item, float) for item in value):
            baseline = EXNRG[key][0]
            for i in range(len(EXNRG[key])):
                EXNRG[key][i] = EXNRG[key][i] - baseline
        # If the list contains missing energies, it finds the first energy to use
        # as baseline and subtracts that value only from the other energies in the list.
        else:
            baseline = 0
            for item in value:
                if isinstance(item, float):
                    baseline = item
                    break
                
            for i in range(len(EXNRG[key])):
                if isinstance(EXNRG[key][i], float):
                    EXNRG[key][i] = EXNRG[key][i] - baseline
                    
    # EXNRG: A dictionary of experimental energies.
    # THNRG: A dictionary of theoretical energies.
    #
    # Each dictionary maps a term symbol (key) to a list of energy levels (value).
    # Example item: '7F2': [0.0, 6.075166999999993, 12.683460900000014, 22.2797451]
     
    if not Ar_dict and type(symmetry) == str:
        Ar_dict = YZS_lib.allowed_cfp(symmetry)
        
    term_to_fit = []
    if not alt_thet_enabled:
        for IT, OEF in YZS_lib.Thet.items():
            if IT[0].upper() == ion and IT[1] in EXNRG:
                term_to_fit.append(IT[1])
    else:
        for IT, OEF in YZS_lib.alt_Thet.items():
            if IT[0].upper() == ion and IT[1] in EXNRG:
                term_to_fit.append(IT[1])
    if len(term_to_fit) == 0:
        print('No matching term symbols found for fitting. Exiting.')
        sys.exit()
    else:
        print('The following term symbols will be fitted:')
        print(term_to_fit)        
    term_not_to_fit = [] 
    for key in EXNRG:
        if key not in term_to_fit:
            term_not_to_fit.append(key)
    if len(term_not_to_fit) > 0:
        print("The following term symbols will not be fitted:")
        print(term_not_to_fit)
        
    #Format of individual bounds: [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-100.0, 100.0), (-10.0, 10.0)]
    assert isinstance(individual_bounds, list)
    assert all(isinstance(item, (tuple, type(None))) for item in individual_bounds)
    if all(isinstance(item, tuple) for item in individual_bounds):
        assert all(isinstance(item[0], (float, type(None))) and
                   isinstance(item[1], (float, type(None)))
                   for item in individual_bounds)
        for item in individual_bounds:
            if isinstance(item[0], float) and isinstance(item[1], float):
                if item[0] >= item[1]:
                    print("Invalid individual bound: ")
                    print(item)
                    raise ValueError
                
    
    
    if len(individual_bounds) == 0:
        individual_bounds = [None] * len(Ar_dict)
    
    # 'shared_bounds' is used as a fallback when a specific bound is missing from 'individual_bounds'.
    # Example format: (-200.0, 200.0)
    # If 'shared_bounds' is not provided, a hardcoded default is used instead.
    # The `individual_bounds` variable is compatible with `scipy.optimize.minimize`.
    for i in range(len(individual_bounds)):
        if isinstance(individual_bounds[i], tuple):
            continue
        elif isinstance(shared_bounds, tuple):
            individual_bounds[i] = shared_bounds
        else:
            #The author is also not confident about this.
            individual_bounds[i] = (-250.0, 250.0)
    assert len(Ar_dict) == len(individual_bounds)
    
    print('Ion: ' + ion)
    print('Energy unit: mev')
        
        
    if symmetry in ['OH', 'O', 'TD']:
        # - p[1] = 5 * p[0]
        # - p[3] = -21 * p[2]
        constraints = {
            1: (0, 5),
            3: (2, -21)
        }
    elif symmetry in ['T', 'TH']:
        # - p[1] = 5 * p[0]
        # - p[3] = -21 * p[2]
        # - p[5] = -1 * p[4]
        constraints = {
            1: (0, 5),
            3: (2, -21),
            5: (4, -1)
        }
    else:
        constraints = {}
    
    with Pool(processes = cpu_count()) as pool:
        args_for_worker = [(
            time,                 #for run_duration_minutes
            individual_bounds,    #for individual_bounds
            constraints,          #constraints
            YZS_lib.core_func,    #for core_func
            EXNRG,                # extra arg for core_func  
            ion,                  # extra arg for core_func
            Ar_dict.copy(),       # extra arg for core_func
            term_to_fit,          # extra arg for core_func
            alt_thet_enabled      # extra arg for core_func
            ) for _ in range(cpu_count())]
        global_results = pool.starmap(YZS_lib.worker, args_for_worker)
        
   
    best_result_tuple = min(global_results, key=lambda x: x[0])
    
    indices_to_remove = set(constraints.keys())
    filtered_bounds = [bound for i, bound in enumerate(individual_bounds) if i not in indices_to_remove]
    
    final_result = minimize(
        fun=YZS_lib.reconstruct_and_run,
        x0=best_result_tuple[1],
        args=(
            constraints,
            YZS_lib.core_func,
            EXNRG,
            ion,
            Ar_dict,
            term_to_fit,
            alt_thet_enabled
        ),
        method='Powell',  
        bounds = filtered_bounds)
 
    
    print(' ')
    print('==================================================================')
    print('*************************Resuls***********************************')
    # print to user
    
    total_vars = len(final_result.x) + len(constraints)
    p_full = np.zeros(total_vars)
    ind_indices = sorted(list(set(range(total_vars)) - set(constraints.keys())))
    p_full[ind_indices] = final_result.x
    for dep, (ind, ratio) in constraints.items():
        p_full[dep] = p_full[ind] * ratio
    
    
    best_THNRG = YZS_lib.THNRG(ion = ion, Ar_dict = Ar_dict, p = p_full, term_to_fit = term_to_fit, alt_thet_enabled = False)
    
    best_Ar = Ar_dict.copy()
    assert len(best_Ar) == len(p_full)
    for key, new_value in zip(best_Ar.keys(), p_full):
        best_Ar[key] = new_value
    new_dictionary = {key[0] + key[1].lower() + key[2:] if len(key) > 1 else key: value
                  for key, value in best_Ar.items()}
    best_Ar = new_dictionary
    print(' ')
    print("Fitted Crystal Field Parameters:")
    for key, value in best_Ar.items():
        print(f"{key}: {value}")
        
    print(' ')
    print(f"Sum of Squared Differences: {final_result.fun}")
    
    print(' ')
    print('Comparison between theoretical and experimental energies: ')
    print('Theoretical      Experimental')
    
    for key in term_to_fit:
        
        assert len(EXNRG[key]) == len(best_THNRG[key])
        
        if not all(isinstance(item, float) for item in EXNRG[key]):
            for i in range(len(EXNRG[key])):
                best_THNRG[key].sort()
                if isinstance(EXNRG[key][i], float):
                    baseline_th = best_THNRG[key][i]
                    break
            for i in range(len(EXNRG[key])):
                best_THNRG[key][i] = best_THNRG[key][i] - baseline_th
            
        
        
        print(key)
        for i in range(len(EXNRG[key])):
            if isinstance(EXNRG[key][i], float):
                print(f"{best_THNRG[key][i]:10.6f}  :  {EXNRG[key][i]:10.6f}")
            elif isinstance(EXNRG[key][i], str):
                print(f"{best_THNRG[key][i]:10.6f}  :   None")
            else:
                raise ValueError
        print(' ')
        
    YZS_lib.create_energy_level_plot(best_THNRG, EXNRG)
        
    if sys.platform == "win32":
        import winsound
        winsound.Beep(15000, 4000)
        winsound.Beep(440, 40000)
    
    