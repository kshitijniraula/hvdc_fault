clear all; close all; clc;

data_path = 'raw_hvdc_fault_data.mat';

% Configuration Parameters
sampling_frequency = 135000;
window_duration_ms = 0.5;
window_size_samples = round(sampling_frequency * (window_duration_ms / 1000));
mother_wavelet = 'db4';
decomposition_level = 4;

samples_per_ms = sampling_frequency / 1000;

% Measurement Variables (DC only)
measurement_vars_to_analyze = {
    'Vdc_T1_Pos', 'Vdc_T1_Neg', 'Idc_T1_Pos', 'Idc_T1_Neg', ...
    'Vdc_T2_Pos', 'Vdc_T2_Neg', 'Idc_T2_Pos', 'Idc_T2_Neg', ...
    'Vdc_T3_Pos', 'Vdc_T3_Neg', 'Idc_T3_Pos', 'Idc_T3_Neg'
};

loaded_data = load(data_path);
sim_runs = loaded_data.final_results_collection;

all_features_matrix = []; % Rows = samples, Cols = features
all_labels_numeric = []; % Numeric labels directly for training 
all_window_end_times = []; % Time corresponding to each feature vector

num_features_per_signal_type = decomposition_level + 1 + 1;
total_features_per_sim = length(measurement_vars_to_analyze) * num_features_per_signal_type;

% class mapping for 8-Class Categorization
label_to_numeric_map = containers.Map('KeyType','char','ValueType','double');
label_to_numeric_map('NoFault') = 0;
label_to_numeric_map('Pole1_Pos_Fault') = 1;
label_to_numeric_map('Pole1_Neg_Fault') = 2;
label_to_numeric_map('Pole2_Pos_Fault') = 3;
label_to_numeric_map('Pole2_Neg_Fault') = 4;
label_to_numeric_map('Pole3_Pos_Fault') = 5;
label_to_numeric_map('Pole3_Neg_Fault') = 6;
label_to_numeric_map('Pole_to_Pole_Fault') = 7;

total_simulations_to_process = length(sim_runs);
overall_start_time = tic;

% Main Loop
for i = 1:total_simulations_to_process

    current_sim_local = sim_runs{i};

    if isfield(current_sim_local, 'Error') && ~isempty(current_sim_local.Error)
        continue;
    end

    signal_data_struct_local = struct();
    all_signals_valid_local = true;
    for j = 1:length(measurement_vars_to_analyze)
        signal_name_local = measurement_vars_to_analyze{j};
        current_signal_local = double(current_sim_local.(signal_name_local));
        if isrow(current_signal_local)
            current_signal_local = current_signal_local';
        end
        signal_data_struct_local.(signal_name_local) = current_signal_local;
    end

    if ~all_signals_valid_local
        continue;
    end

    time_vector_local = current_sim_local.Time;
    fault_inception_time_local = current_sim_local.FaultInceptionTime;
    current_fault_type_string_local = current_sim_local.FaultType;

     % Determining the true numeric label for the current simulation's fault type 
    if isKey(label_to_numeric_map, current_fault_type_string_local)
        current_numeric_fault_label_local = label_to_numeric_map(current_fault_type_string_local);
    elseif strcmp(current_fault_type_string_local, 'f_1p')
        current_numeric_fault_label_local = label_to_numeric_map('Pole1_Pos_Fault');
    elseif strcmp(current_fault_type_string_local, 'f_1n')
        current_numeric_fault_label_local = label_to_numeric_map('Pole1_Neg_Fault');
    elseif strcmp(current_fault_type_string_local, 'f_2p')
        current_numeric_fault_label_local = label_to_numeric_map('Pole2_Pos_Fault');
    elseif strcmp(current_fault_type_string_local, 'f_2n')
        current_numeric_fault_label_local = label_to_numeric_map('Pole2_Neg_Fault');
    elseif strcmp(current_fault_type_string_local, 'f_3p')
        current_numeric_fault_label_local = label_to_numeric_map('Pole3_Pos_Fault');
    elseif strcmp(current_fault_type_string_local, 'f_3n')
        current_numeric_fault_label_local = label_to_numeric_map('Pole3_Neg_Fault');
    elseif strcmp(current_fault_type_string_local, 'f_pp')
        current_numeric_fault_label_local = label_to_numeric_map('Pole_to_Pole_Fault');
    else
        current_numeric_fault_label_local = label_to_numeric_map('NoFault');
    end

    % Initializing local arrays
    features_this_sim = [];
    labels_this_sim = [];
    times_this_sim = [];

    % No fault for first 20ms
    initial_stabilization_duration_ms = 20;
    num_initial_stabilization_samples = initial_stabilization_duration_ms * samples_per_ms;

    % Is there enough samples for at least one full window in the stabilization period?
    if num_initial_stabilization_samples >= window_size_samples %yes
        num_windows_for_initial_stabilization = 10;
        candidate_start_indices_stabilization = 1 : (num_initial_stabilization_samples - window_size_samples + 1);

        % select 10 evenly spaced windows within this initial stabilization period
        if length(candidate_start_indices_stabilization) > 0
            if length(candidate_start_indices_stabilization) <= num_windows_for_initial_stabilization
                windows_to_extract_stabilization = candidate_start_indices_stabilization;
            else
                windows_to_extract_stabilization = round(linspace(candidate_start_indices_stabilization(1), candidate_start_indices_stabilization(end), num_windows_for_initial_stabilization));
            end

            for start_idx_stabilization = windows_to_extract_stabilization
                end_idx_stabilization = start_idx_stabilization + window_size_samples - 1;
                window_end_time_stabilization = time_vector_local(end_idx_stabilization);

                features_for_this_window_local = zeros(1, total_features_per_sim);
                current_feature_output_idx = 1;

                for j_local = 1:length(measurement_vars_to_analyze)
                    signal_name_local_inner = measurement_vars_to_analyze{j_local};
                    analysis_window_local = signal_data_struct_local.(signal_name_local_inner)(start_idx_stabilization:end_idx_stabilization);

                    if any(isnan(analysis_window_local)) || any(isinf(analysis_window_local)) || isempty(analysis_window_local) || all(analysis_window_local == 0)
                         features_for_this_window_local(current_feature_output_idx : current_feature_output_idx + num_features_per_signal_type - 1) = nan;
                         current_feature_output_idx = current_feature_output_idx + num_features_per_signal_type;
                         continue;
                    end

                    % Inlined Feature Extraction Logic
                    num_features_per_single_signal_inline = decomposition_level + 1 + 1;

                    if size(analysis_window_local, 2) > 1
                        analysis_window_local = analysis_window_local';
                    end
                    if isempty(analysis_window_local) || all(analysis_window_local == 0) || any(isnan(analysis_window_local)) || any(isinf(analysis_window_local))
                        current_signal_features_inline = NaN(1, num_features_per_single_signal_inline);
                    else
                        [C_inline, L_inline] = wavedec(analysis_window_local, decomposition_level, mother_wavelet);
                        energies_for_entropy_inline = zeros(1, decomposition_level + 1);
                        energy_idx_inline = 1;
                        current_feature_inline_idx = 1;
                        current_signal_features_inline = zeros(1, num_features_per_single_signal_inline);
                        for level_inline = 1:decomposition_level
                            cD_inline = detcoef(C_inline, L_inline, level_inline);
                            current_signal_features_inline(current_feature_inline_idx) = rms(cD_inline);
                            current_feature_inline_idx = current_feature_inline_idx + 1;
                            energies_for_entropy_inline(energy_idx_inline) = sum(cD_inline.^2);
                            energy_idx_inline = energy_idx_inline + 1;
                        end
                        cA_last_level_val_inline = appcoef(C_inline, L_inline, mother_wavelet, decomposition_level);
                        current_signal_features_inline(current_feature_inline_idx) = rms(cA_last_level_val_inline);
                        current_feature_inline_idx = current_feature_inline_idx + 1;
                        energies_for_entropy_inline = [energies_for_entropy_inline, sum(cA_last_level_val_inline.^2)];
                        energy_idx_inline = energy_idx_inline + 1;
                        total_energy_for_entropy_inline = sum(energies_for_entropy_inline);
                        if total_energy_for_entropy_inline == 0
                            entropy_val_inline = 0;
                        else
                            p_energies_inline = energies_for_entropy_inline / total_energy_for_entropy_inline;
                            p_energies_nonzero_inline = p_energies_inline(p_energies_inline > 0);
                            if isempty(p_energies_nonzero_inline)
                                entropy_val_inline = 0;
                            else
                                entropy_val_inline = -sum(p_energies_nonzero_inline .* log2(p_energies_nonzero_inline), 'omitnan');
                            end
                        end
                        current_signal_features_inline(current_feature_inline_idx) = entropy_val_inline;
                    end

                    features_for_this_window_local(current_feature_output_idx : current_feature_output_idx + num_features_per_signal_type - 1) = current_signal_features_inline;
                    current_feature_output_idx = current_feature_output_idx + num_features_per_signal_type;

                end

                % Label as NoFault
                label_for_this_window_local_stabilization = label_to_numeric_map('NoFault');
                features_this_sim = [features_this_sim; features_for_this_window_local];
                labels_this_sim = [labels_this_sim; label_for_this_window_local_stabilization];
                times_this_sim = [times_this_sim; window_end_time_stabilization];
            end
        end
    end

    % Targeted Window Selection Logic
    windows_to_extract_indices = [];
    num_total_samples_in_sim = length(time_vector_local);

    if strcmp(current_fault_type_string_local, 'NoFault')
        % For NoFault, extract 10 evenly spaced windows across the entire simulation duration
        num_windows_for_no_fault = 10;
        if num_total_samples_in_sim < window_size_samples
            windows_to_extract_indices = [];
        else
            candidate_start_indices = 1 : (num_total_samples_in_sim - window_size_samples + 1);
            
            candidate_start_indices = candidate_start_indices(candidate_start_indices > num_initial_stabilization_samples);
            if isempty(candidate_start_indices)
                windows_to_extract_indices = [];
            elseif length(candidate_start_indices) <= num_windows_for_no_fault
                windows_to_extract_indices = candidate_start_indices;
            else
                windows_to_extract_indices = round(linspace(candidate_start_indices(1), candidate_start_indices(end), num_windows_for_no_fault));
            end
        end
    else % For faulted simulations, capturing around 100 window
        num_windows_for_fault = 100;

        idx_inception_sample = find(time_vector_local >= fault_inception_time_local, 1, 'first');
        if isempty(idx_inception_sample)
            idx_inception_sample = round(num_total_samples_in_sim / 2);
        end

        % Analysis zone (from 5ms before to 50ms after)
        analysis_zone_start_sample = max(1, idx_inception_sample - 5 * samples_per_ms);
        analysis_zone_end_sample = min(num_total_samples_in_sim - window_size_samples + 1, idx_inception_sample + 50 * samples_per_ms);

        % Generate candidate window start indices within this zone
        candidate_indices_in_zone = (analysis_zone_start_sample : analysis_zone_end_sample);
        candidate_indices_in_zone = candidate_indices_in_zone(candidate_indices_in_zone > num_initial_stabilization_samples);
        if isempty(candidate_indices_in_zone)
            mid_sim_idx = round(num_total_samples_in_sim / 2);
            windows_to_extract_indices = max(1, min(mid_sim_idx, num_total_samples_in_sim - window_size_samples + 1));
        elseif length(candidate_indices_in_zone) <= num_windows_for_fault
            windows_to_extract_indices = candidate_indices_in_zone;
        else
            windows_to_extract_indices = round(linspace(candidate_indices_in_zone(1), candidate_indices_in_zone(end), num_windows_for_fault));
        end
    end

    for start_idx_local = windows_to_extract_indices
        end_idx_local = start_idx_local + window_size_samples - 1;
        window_end_time_local = time_vector_local(end_idx_local);
        features_for_this_window_local = zeros(1, total_features_per_sim);
        current_feature_output_idx = 1;
        for j_local = 1:length(measurement_vars_to_analyze)
            signal_name_local_inner = measurement_vars_to_analyze{j_local};
            analysis_window_local = signal_data_struct_local.(signal_name_local_inner)(start_idx_local:end_idx_local);
            if any(isnan(analysis_window_local)) || any(isinf(analysis_window_local)) || isempty(analysis_window_local) || all(analysis_window_local == 0)
                 features_for_this_window_local(current_feature_output_idx : current_feature_output_idx + num_features_per_signal_type - 1) = nan;
                 current_feature_output_idx = current_feature_output_idx + num_features_per_signal_type;
                 continue;
            end
            num_features_per_single_signal_inline = decomposition_level + 1 + 1;
            if size(analysis_window_local, 2) > 1
                analysis_window_local = analysis_window_local';
            end
            if isempty(analysis_window_local) || all(analysis_window_local == 0) || any(isnan(analysis_window_local)) || any(isinf(analysis_window_local))
                current_signal_features_inline = NaN(1, num_features_per_single_signal_inline);
            else
                [C_inline, L_inline] = wavedec(analysis_window_local, decomposition_level, mother_wavelet);
                energies_for_entropy_inline = zeros(1, decomposition_level + 1);
                energy_idx_inline = 1;
                current_feature_inline_idx = 1;
                current_signal_features_inline = zeros(1, num_features_per_single_signal_inline);
                for level_inline = 1:decomposition_level
                    cD_inline = detcoef(C_inline, L_inline, level_inline);
                    current_signal_features_inline(current_feature_inline_idx) = rms(cD_inline);
                    current_feature_inline_idx = current_feature_inline_idx + 1;
                    energies_for_entropy_inline(energy_idx_inline) = sum(cD_inline.^2);
                    energy_idx_inline = energy_idx_inline + 1;
                end
                cA_last_level_val_inline = appcoef(C_inline, L_inline, mother_wavelet, decomposition_level);
                current_signal_features_inline(current_feature_inline_idx) = rms(cA_last_level_val_inline);
                current_feature_inline_idx = current_feature_inline_idx + 1;
                energies_for_entropy_inline = [energies_for_entropy_inline, sum(cA_last_level_val_inline.^2)];
                energy_idx_inline = energy_idx_inline + 1;
                total_energy_for_entropy_inline = sum(energies_for_entropy_inline);
                if total_energy_for_entropy_inline == 0
                    entropy_val_inline = 0;
                else
                    p_energies_inline = energies_for_entropy_inline / total_energy_for_entropy_inline;
                    p_energies_nonzero_inline = p_energies_inline(p_energies_inline > 0);
                    if isempty(p_energies_nonzero_inline)
                        entropy_val_inline = 0;
                    else
                        entropy_val_inline = -sum(p_energies_nonzero_inline .* log2(p_energies_nonzero_inline), 'omitnan');
                    end
                end
                current_signal_features_inline(current_feature_inline_idx) = entropy_val_inline;
            end
            features_for_this_window_local(current_feature_output_idx : current_feature_output_idx + num_features_per_single_signal_inline - 1) = current_signal_features_inline;
            current_feature_output_idx = current_feature_output_idx + num_features_per_single_signal_inline;
        end

        if strcmp(current_fault_type_string_local, 'NoFault')
             label_for_this_window_local = label_to_numeric_map('NoFault');
        elseif window_end_time_local >= fault_inception_time_local
             label_for_this_window_local = current_numeric_fault_label_local;
        else
             label_for_this_window_local = label_to_numeric_map('NoFault');
        end
        features_this_sim = [features_this_sim; features_for_this_window_local];
        labels_this_sim = [labels_this_sim; label_for_this_window_local];
        times_this_sim = [times_this_sim; window_end_time_local];
    end

    all_features_matrix = [all_features_matrix; features_this_sim];
    all_labels_numeric = [all_labels_numeric; labels_this_sim];
    all_window_end_times = [all_window_end_times; times_this_sim];
end

% Feature names based on the 6 features per signal
feature_names_list = {};
for j = 1:length(measurement_vars_to_analyze)
    signal_name_prefix = measurement_vars_to_analyze{j};
    for level = 1:decomposition_level
        feature_names_list{end+1} = sprintf('%s_cD%d_RMS', signal_name_prefix, level);
    end
    feature_names_list{end+1} = sprintf('%s_cA%d_RMS', signal_name_prefix, decomposition_level);
    feature_names_list{end+1} = sprintf('%s_Wavelet_Entropy', signal_name_prefix);
end
json_output_file = 'metadata_features_labels.json';
metadata_to_json.label_to_numeric = struct();
keys_map = label_to_numeric_map.keys;
values_map = label_to_numeric_map.values;
for k = 1:length(keys_map)
    metadata_to_json.label_to_numeric.(matlab.lang.makeValidName(keys_map{k})) = values_map{k};
end
metadata_to_json.feature_names = feature_names_list;
if exist('jsonencode', 'builtin')
    fid = fopen(json_output_file, 'w');
    fprintf(fid, '%s', jsonencode(metadata_to_json));
    fclose(fid);
end
feature_data_output_file = 'extracted_wavelet_features.mat';
save(feature_data_output_file, ...
    'all_features_matrix', ...
    'all_labels_numeric', ...
    'all_window_end_times', ...
    'feature_names_list', ...
    '-v7.3');