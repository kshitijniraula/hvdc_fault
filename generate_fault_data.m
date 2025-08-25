clear all; close all; clc;

model_name = 'HVDCVSCBased2023_RawDataGen';
simulation_duration = 0.5;
noise_level_percentage = 0.015;

fault_types = {
    'NoFault', 'f_1p', 'f_1n', 'f_2p', 'f_2n', 'f_3p', 'f_3n', 'f_pp'
};

fault_inception_times = 0.05:0.005:0.45;

fault_resistances = 0.001;

data_file_to_save = 'raw_hvdc_fault_data.mat';
save_interval = 1;
resume_from_last_save = true;
all_sim_results = {};
scenario_idx = 0;
initial_time_vector = [0; 1];
initial_value_vector = [0; 0];

fault_f1p_control = [initial_time_vector, initial_value_vector];
fault_f1n_control = [initial_time_vector, initial_value_vector];
fault_f2p_control = [initial_time_vector, initial_value_vector];
fault_f2n_control = [initial_time_vector, initial_value_vector];
fault_f3p_control = [initial_time_vector, initial_value_vector];
fault_f3n_control = [initial_time_vector, initial_value_vector];
fault_pp_control = [initial_time_vector, initial_value_vector];
FAULT_INCEPTION_TIME = 1e9;

sim_time = [];
Vdc_T1_Pos = []; Vdc_T1_Neg = []; Idc_T1_Pos = []; Idc_T1_Neg = [];
Vdc_T2_Pos = []; Vdc_T2_Neg = []; Idc_T2_Pos = []; Idc_T2_Neg = [];
Vdc_T3_Pos = []; Vdc_T3_Neg = []; Idc_T3_Pos = []; Idc_T3_Neg = [];
Vac_C1_A = []; Vac_C1_B = []; Vac_C1_C = []; Iac_C1_A = []; Iac_C1_B = []; Iac_C1_C = [];
Vac_C2_A = []; Vac_C2_B = []; Vac_C2_C = []; Iac_C2_A = []; Iac_C2_B = []; Iac_C2_C = [];

load_system(model_name);
set_param(model_name, 'SimulationCommand', 'update');

if ~verLessThan('matlab', '8.0') && license('test', 'Distrib_Computing_Toolbox')
    if isempty(gcp('nocreate'))
        parpool('local', 16);
    end
    use_parfor = true;
else
    use_parfor = false;
end

all_scenario_params = {};

num_no_fault_scenarios_initial = 100;
for i = 1:num_no_fault_scenarios_initial
    scenario_idx = scenario_idx + 1;
    all_scenario_params{scenario_idx} = struct(...
        'FaultType', 'NoFault', ...
        'FaultLocation', 'N/A', ...
        'FaultResistance', 'N/A', ...
        'FaultInceptionTime', simulation_duration + 1, ...
        'OperatingCondition', sprintf('FixedOperatingPoint_NF%d', i), ...
        'SimID', scenario_idx);
end

for f_type_idx = 1:length(fault_types)
    current_fault_type = fault_types{f_type_idx};
    if strcmp(current_fault_type, 'NoFault')
        continue;
    end
    switch current_fault_type
        case 'f_1p'
            current_fault_location = 'Terminal1_PosPole';
        case 'f_1n'
            current_fault_location = 'Terminal1_NegPole';
        case 'f_2p'
            current_fault_location = 'Terminal2_PosPole';
        case 'f_2n'
            current_fault_location = 'Terminal2_NegPole';
        case 'f_3p'
            current_fault_location = 'Terminal3_PosPole';
        case 'f_3n'
            current_fault_location = 'Terminal3_NegPole';
        case 'f_pp'
            current_fault_location = 'BranchLine_Mid';
        otherwise
            current_fault_location = 'Unknown_Location';
    end

    for r_idx = 1:length(fault_resistances)
        current_fault_resistance_val = fault_resistances(r_idx);

        for inc_time_idx = 1:length(fault_inception_times)
            current_fault_inception_time = fault_inception_times(inc_time_idx);

            current_op_cond = 'FixedOperatingPoint';
            scenario_idx = scenario_idx + 1;
            all_scenario_params{scenario_idx} = struct(...
                'FaultType', current_fault_type, ...
                'FaultLocation', current_fault_location, ...
                'FaultResistance', current_fault_resistance_val, ...
                'FaultInceptionTime', current_fault_inception_time, ...
                'OperatingCondition', current_op_cond, ...
                'SimID', scenario_idx);
        end
    end
end

start_sim_idx = 1;
if resume_from_last_save && exist(data_file_to_save, 'file')
    loaded_data = load(data_file_to_save);
    if isfield(loaded_data, 'final_results_collection') && ~isempty(loaded_data.final_results_collection)
        final_results_collection = loaded_data.final_results_collection;
        start_sim_idx = length(final_results_collection) + 1;
        for i = 1:(start_sim_idx - 1)
            if ~isfield(final_results_collection{i}, 'SimID')
                final_results_collection{i}.SimID = i;
            end
        end
    else
        final_results_collection = cell(1, length(all_scenario_params));
    end
else
    final_results_collection = cell(1, length(all_scenario_params));
end

num_sims_to_run_in_batch = length(all_scenario_params) - start_sim_idx + 1;

if use_parfor
    current_batch_results = cell(1, num_sims_to_run_in_batch);
    parfor batch_idx = 1:num_sims_to_run_in_batch
        sim_idx_global = start_sim_idx + batch_idx - 1;
        worker_model_name = model_name;
        load_system(worker_model_name);
        current_scenario = all_scenario_params{sim_idx_global};
        try
            sim_result_struct = run_single_simulation(...
                worker_model_name, ...
                simulation_duration, ...
                current_scenario.FaultType, ...
                current_scenario.FaultLocation, ...
                current_scenario.FaultResistance, ...
                current_scenario.FaultInceptionTime, ...
                current_scenario.OperatingCondition, ...
                noise_level_percentage, ...
                current_scenario.SimID);
        catch ME
            sim_result_struct = struct();
            sim_result_struct.SimID = current_scenario.SimID;
            sim_result_struct.Error = ME.message;
            measurement_vars = {
                'Vdc_T1_Pos', 'Vdc_T1_Neg', 'Idc_T1_Pos', 'Idc_T1_Neg', ...
                'Vdc_T2_Pos', 'Vdc_T2_Neg', 'Idc_T2_Pos', 'Idc_T2_Neg', ...
                'Vdc_T3_Pos', 'Vdc_T3_Neg', 'Idc_T3_Pos', 'Idc_T3_Neg', ...
                'Vac_C1_A', 'Vac_C1_B', 'Vac_C1_C', 'Iac_C1_A', 'Iac_C1_B', 'Iac_C1_C', ...
                'Vac_C2_A', 'Vac_C2_B', 'Vac_C2_C', 'Iac_C2_A', 'Iac_C2_B', 'Iac_C2_C'
            };
            for m_var_idx = 1:length(measurement_vars)
                sim_result_struct.(measurement_vars{m_var_idx}) = [];
            end
            sim_result_struct.Time = [];
            sim_result_struct.FaultType = current_scenario.FaultType;
            sim_result_struct.FaultLocation = current_scenario.FaultLocation;
            sim_result_struct.FaultResistance = current_scenario.FaultResistance;
            sim_result_struct.FaultInceptionTime = current_scenario.FaultInceptionTime;
            sim_result_struct.OperatingCondition = current_scenario.OperatingCondition;
        end
        current_batch_results{batch_idx} = sim_result_struct;
        close_system(worker_model_name, 0);
    end
    final_results_collection(start_sim_idx:length(all_scenario_params)) = current_batch_results;
    save(data_file_to_save, 'final_results_collection', '-v7.3');
else
    for sim_idx = start_sim_idx:length(all_scenario_params)
        current_scenario = all_scenario_params{sim_idx};
        try
            sim_result_struct = run_single_simulation(...
                model_name, ...
                simulation_duration, ...
                current_scenario.FaultType, ...
                current_scenario.FaultLocation, ...
                current_scenario.FaultResistance, ...
                current_scenario.FaultInceptionTime, ...
                current_scenario.OperatingCondition, ...
                noise_level_percentage, ...
                current_scenario.SimID);
        catch ME
            sim_result_struct = struct();
            sim_result_struct.SimID = current_scenario.SimID;
            sim_result_struct.Error = ME.message;
            measurement_vars = {
                'Vdc_T1_Pos', 'Vdc_T1_Neg', 'Idc_T1_Pos', 'Idc_T1_Neg', ...
                'Vdc_T2_Pos', 'Vdc_T2_Neg', 'Idc_T2_Pos', 'Idc_T2_Neg', ...
                'Vdc_T3_Pos', 'Vdc_T3_Neg', 'Idc_T3_Pos', 'Idc_T3_Neg', ...
                'Vac_C1_A', 'Vac_C1_B', 'Vac_C1_C', 'Iac_C1_A', 'Iac_C1_B', 'Iac_C1_C', ...
                'Vac_C2_A', 'Vac_C2_B', 'Vac_C2_C', 'Iac_C2_A', 'Iac_C2_B', 'Iac_C2_C'
            };
            for m_var_idx = 1:length(measurement_vars)
                sim_result_struct.(measurement_vars{m_var_idx}) = [];
            end
            sim_result_struct.Time = [];
            sim_result_struct.FaultType = current_scenario.FaultType;
            sim_result_struct.FaultLocation = current_scenario.FaultLocation;
            sim_result_struct.FaultResistance = current_scenario.FaultResistance;
            sim_result_struct.FaultInceptionTime = current_scenario.FaultInceptionTime;
            sim_result_struct.OperatingCondition = current_scenario.OperatingCondition;
        end
        final_results_collection{sim_idx} = sim_result_struct;
        if mod(sim_idx, save_interval) == 0 || sim_idx == length(all_scenario_params)
            save(data_file_to_save, 'final_results_collection', '-v7.3');
        end
    end
end
save(data_file_to_save, 'final_results_collection', '-v7.3');
if ~use_parfor
    close_system(model_name, 0);
end
clear_vars_for_cleanup = [
    {'FAULT_INCEPTION_TIME', 'sim_time'}, ...
    {'fault_f1p_control', 'fault_f1n_control', 'fault_f2p_control', ...
     'fault_f2n_control', 'fault_f3p_control', 'fault_f3n_control', ...
     'fault_pp_control'}, ...
    {'Vdc_T1_Pos', 'Vdc_T1_Neg', 'Idc_T1_Pos', 'Idc_T1_Neg', ...
     'Vdc_T2_Pos', 'Vdc_T2_Neg', 'Idc_T2_Pos', 'Idc_T2_Neg', ...
     'Vdc_T3_Pos', 'Vdc_T3_Neg', 'Idc_T3_Pos', 'Idc_T3_Neg', ...
     'Vac_C1_A', 'Vac_C1_B', 'Vac_C1_C', 'Iac_C1_A', 'Iac_C1_B', 'Iac_C1_C', ...
     'Vac_C2_A', 'Vac_C2_B', 'Vac_C2_C', 'Iac_C2_A', 'Iac_C2_B', 'Iac_C2_C'}, ...
    {'all_scenario_params', 'scenario_idx', 'use_parfor', 'model_name', 'simulation_duration', 'noise_level_percentage', 'fault_types', 'num_no_fault_scenarios_initial', 'fault_inception_times', 'fault_resistances', 'start_sim_idx', 'current_batch_results', 'save_interval', 'resume_from_last_save', 'data_file_to_save', 'num_sims_to_run_in_batch'}
];
clear(clear_vars_for_cleanup{:});
if ~isempty(gcp('nocreate'))
    delete(gcp('nocreate'));
end
function result_struct = run_single_simulation(model_name_local, sim_duration_local, ...
        fault_type_local, fault_location_local, fault_resistance_local_value, fault_inception_time_local_value, ...
        operating_condition_local, noise_level_percentage_local, sim_id_local)
    off_signal = [0, 0; sim_duration_local, 0];
    assignin('base', 'fault_f1p_control', off_signal);
    assignin('base', 'fault_f1n_control', off_signal);
    assignin('base', 'fault_f2p_control', off_signal);
    assignin('base', 'fault_f2n_control', off_signal);
    assignin('base', 'fault_f3p_control', off_signal);
    assignin('base', 'fault_f3n_control', off_signal);
    assignin('base', 'fault_pp_control', off_signal);

    assignin('base', 'FAULT_INCEPTION_TIME', sim_duration_local + 1);

    if ~strcmp(fault_type_local, 'NoFault')
        epsilon = 1e-9;
        fault_signal_time = [0; fault_inception_time_local_value - epsilon; fault_inception_time_local_value; sim_duration_local];
        fault_signal_value = [0; 0; 1; 1];
        fault_time_series_data = [fault_signal_time, fault_signal_value];
        switch fault_type_local
            case 'f_1p'
                assignin('base', 'fault_f1p_control', fault_time_series_data);
            case 'f_1n'
                assignin('base', 'fault_f1n_control', fault_time_series_data);
            case 'f_2p'
                assignin('base', 'fault_f2p_control', fault_time_series_data);
            case 'f_2n'
                assignin('base', 'fault_f2n_control', fault_time_series_data);
            case 'f_3p'
                assignin('base', 'fault_f3p_control', fault_time_series_data);
            case 'f_3n'
                assignin('base', 'fault_f3n_control', fault_time_series_data);
            case 'f_pp'
                assignin('base', 'fault_pp_control', fault_time_series_data);
            otherwise
                assignin('base', 'fault_f1p_control', off_signal); assignin('base', 'fault_f1n_control', off_signal);
                assignin('base', 'fault_f2p_control', off_signal); assignin('base', 'fault_f2n_control', off_signal);
                assignin('base', 'fault_f3p_control', off_signal); assignin('base', 'fault_f3n_control', off_signal);
                assignin('base', 'fault_pp_control', off_signal);
        end
        assignin('base', 'FAULT_INCEPTION_TIME', fault_inception_time_local_value);
    end
    set_param(model_name_local, 'StopTime', num2str(sim_duration_local));

    try
        sim_output_workspace = sim(model_name_local, 'ReturnWorkspaceOutputs', 'on');
    catch ME
        result_struct = struct();
        result_struct.SimID = sim_id_local;
        result_struct.Error = ME.message;
        measurement_vars = {
            'Vdc_T1_Pos', 'Vdc_T1_Neg', 'Idc_T1_Pos', 'Idc_T1_Neg', ...
            'Vdc_T2_Pos', 'Vdc_T2_Neg', 'Idc_T2_Pos', 'Idc_T2_Neg', ...
            'Vdc_T3_Pos', 'Vdc_T3_Neg', 'Idc_T3_Pos', 'Idc_T3_Neg', ...
            'Vac_C1_A', 'Vac_C1_B', 'Vac_C1_C', 'Iac_C1_A', 'Iac_C1_B', 'Iac_C1_C', ...
            'Vac_C2_A', 'Vac_C2_B', 'Vac_C2_C', 'Iac_C2_A', 'Iac_C2_B', 'Iac_C2_C'
        };
        for m_var_idx = 1:length(measurement_vars)
            sim_result_struct.(measurement_vars{m_var_idx}) = [];
        end
        sim_result_struct.Time = [];
        sim_result_struct.FaultType = fault_type_local;
        sim_result_struct.FaultLocation = fault_location_local;
        sim_result_struct.FaultResistance = fault_resistance_local_value;
        sim_result_struct.FaultInceptionTime = fault_inception_time_local_value;
        sim_result_struct.OperatingCondition = operating_condition_local;
    end
    result_struct = struct();
    measurement_vars = {
        'Vdc_T1_Pos', 'Vdc_T1_Neg', 'Idc_T1_Pos', 'Idc_T1_Neg', ...
        'Vdc_T2_Pos', 'Vdc_T2_Neg', 'Idc_T2_Pos', 'Idc_T2_Neg', ...
        'Vdc_T3_Pos', 'Vdc_T3_Neg', 'Idc_T3_Pos', 'Idc_T3_Neg', ...
        'Vac_C1_A', 'Vac_C1_B', 'Vac_C1_C', 'Iac_C1_A', 'Iac_C1_B', 'Iac_C1_C', ...
        'Vac_C2_A', 'Vac_C2_B', 'Vac_C2_C', 'Iac_C2_A', 'Iac_C2_B', 'Iac_C2_C'
    };
    if isprop(sim_output_workspace, 'sim_time')
        result_struct.Time = sim_output_workspace.sim_time;
    elseif isprop(sim_output_workspace, 'logsout') && ~isempty(sim_output_workspace.logsout)
        if sim_output_workspace.logsout.find('Name', m_var_name)
            result_struct.Time = sim_output_workspace.logsout.get('sim_time').Values.Data;
        else
            result_struct.Time = [];
        end
    else
        result_struct.Time = [];
    end
    if isempty(result_struct.Time)
        error('Could not obtain simulation time for Sim %d. Ensure "sim_time" ''To Workspace'' block is correctly configured in Simulink.', sim_id_local);
    end
    for m_var_idx = 1:length(measurement_vars)
        m_var_name = measurement_vars{m_var_idx};
        signal_data = [];
        if isprop(sim_output_workspace, m_var_name)
            signal_data = sim_output_workspace.(m_var_name);
        elseif isprop(sim_output_workspace, 'logsout') && ~isempty(sim_output_workspace.logsout)
            if sim_output_workspace.logsout.find('Name', m_var_name)
                signal_data = sim_output_workspace.logsout.get(m_var_name).Values.Data;
            end
        else
            result_struct.(m_var_name) = [];
            continue;
        end
        if isempty(signal_data)
            result_struct.(m_var_name) = [];
        else
            if isa(signal_data, 'Simulink.Timeseries')
                signal_data = signal_data.Data;
            end
            signal_data = double(signal_data);
            if isrow(signal_data)
                signal_data = signal_data';
            end
            max_abs_val = max(abs(signal_data));
            if max_abs_val == 0
                noise_std_dev = noise_level_percentage_local;
            else
                noise_std_dev = noise_level_percentage_local * max_abs_val;
            end
            noise = noise_std_dev * randn(size(signal_data));
            signal_data_noisy = signal_data + noise;
            result_struct.(m_var_name) = signal_data_noisy;
        end
    end
    result_struct.FaultType = fault_type_local;
    result_struct.FaultLocation = fault_location_local;
    result_struct.FaultResistance = fault_resistance_local_value;
    result_struct.FaultInceptionTime = fault_inception_time_local_value;
    result_struct.OperatingCondition = operating_condition_local;
    result_struct.SimID = sim_id_local;
end