clear all; close all; clc;

% Part 1: System Configuration
% Define simulation and feature extraction parameters
sampling_frequency = 135000;
mother_wavelet = 'db4';
decomposition_level = 4;
noise_level_percentage = 0.015;
model_name = 'HVDCVSCBased2023_RawDataGen';
simulation_duration = 0.5;

% DEFINE SEPARATE WINDOW PARAMETERS FOR EACH MODEL
window_duration_ms_catboost = 0.5;
window_duration_ms_xgboost = 0.5; 
window_size_samples_catboost = round(sampling_frequency * (window_duration_ms_catboost / 1000));
window_size_samples_xgboost = round(sampling_frequency * (window_duration_ms_xgboost / 1000));
samples_per_ms = sampling_frequency / 1000;

fixed_fault_type = 'f_1p';
fixed_inception_time = 0.15;


sim_id = 9999;
python_script_name = 'real_time_predict_combined.py';

% DEFINE MEASUREMENT VARIABLES
measurement_vars_to_analyze = {
    'Vdc_T1_Pos', 'Vdc_T1_Neg', 'Idc_T1_Pos', 'Idc_T1_Neg', ...
    'Vdc_T2_Pos', 'Vdc_T2_Neg', 'Idc_T2_Pos', 'Idc_T2_Neg', ...
    'Vdc_T3_Pos', 'Vdc_T3_Neg', 'Idc_T3_Pos', 'Idc_T3_Neg'
};

% Determine the fault location based on type
switch fixed_fault_type
    case 'f_1p'
        fault_location = 'Terminal1_PosPole';
    case 'f_1n'
        fault_location = 'Terminal1_NegPole';
    case 'f_2p'
        fault_location = 'Terminal2_PosPole';
    case 'f_2n'
        fault_location = 'Terminal2_NegPole';
    case 'f_3p'
        fault_location = 'Terminal3_PosPole';
    case 'f_3n'
        fault_location = 'Terminal3_NegPole';
    case 'f_pp'
        fault_location = 'BranchLine_Mid';
    otherwise
        fault_location = 'NoFault';
end

fprintf('1. Simulating a single fault: %s at %.3f s\n', fixed_fault_type, fixed_inception_time);
sim_result = run_single_simulation(...
    model_name, ...
    simulation_duration, ...
    fixed_fault_type, ...
    fault_location, ...
    fixed_inception_time, ...
    'FixedOperatingPoint', ...
    noise_level_percentage, ...
    sim_id);

if isfield(sim_result, 'Error') && ~isempty(sim_result.Error)
    fprintf('Error during simulation: %s\n', sim_result.Error);
    return;
end

fprintf('2. Initializing Python environment...\n');
fprintf('Note: The following step will take a moment as models are loaded into memory for each check.\n');

% Data to be collected for plotting
times_since_inception_cat = [];
catboost_predictions = {};
catboost_is_correct = [];
feature_extraction_times_cat = [];
prediction_times_with_load_cat = [];
prediction_times_only_cat = [];
catboost_false_positives = 0;
detection_time_catboost = nan;

times_since_inception_xgb = [];
xgboost_predictions = {};
xgboost_is_correct = [];
feature_extraction_times_xgb = [];
prediction_times_with_load_xgb = [];
prediction_times_only_xgb = [];
xgboost_false_positives = 0;
detection_time_xgboost = nan;

% Part 3: CATBOOST - Sliding Window Analysis
fprintf('\n3a. Performing continuous analysis for CatBoost (Window: %.1f ms)...\n', window_duration_ms_catboost);
current_window_start_idx = 1;
catboost_correctly_detected = false;

while current_window_start_idx <= length(sim_result.Time) - window_size_samples_catboost
    start_time_extraction = tic;
    
    start_idx = current_window_start_idx;
    end_idx = start_idx + window_size_samples_catboost - 1;
    analysis_time = sim_result.Time(end_idx);

    feature_vector = extract_features_from_window(...
        sim_result, ...
        measurement_vars_to_analyze, ...
        start_idx, ...
        end_idx, ...
        decomposition_level, ...
        mother_wavelet ...
    );
    
    feature_extraction_times_cat(end+1) = toc(start_time_extraction) * 1000;
    
    feature_vector_str = strjoin(cellstr(num2str(feature_vector(:))), ',');
    true_label_str = fixed_fault_type;

    [~, cmdout_predict] = system(sprintf('python "%s" "%s" "%s"', python_script_name, feature_vector_str, true_label_str));
    
    response_predict = jsondecode(cmdout_predict);
    
    if strcmp(response_predict.status, 'error')
        fprintf('Python Prediction Error: %s\n', response_predict.message);
        break;
    end
    
    times_since_inception_cat(end+1) = analysis_time - fixed_inception_time;
    catboost_predictions{end+1} = response_predict.catboost_pred;
    catboost_is_correct(end+1) = response_predict.is_catboost_correct;
    prediction_times_with_load_cat(end+1) = response_predict.load_time_ms + response_predict.prediction_time_ms;
    prediction_times_only_cat(end+1) = response_predict.prediction_time_ms;
    
    idx_inception_sample = find(sim_result.Time >= fixed_inception_time, 1, 'first');
    is_after_inception = (current_window_start_idx >= idx_inception_sample);

    if ~is_after_inception
        if ~strcmp(response_predict.catboost_pred, 'NoFault')
            catboost_false_positives = catboost_false_positives + 1;
        end
    else
        if ~catboost_correctly_detected && catboost_is_correct(end)
            detection_time_catboost = times_since_inception_cat(end);
            catboost_correctly_detected = true;
        end
    end
    
    if catboost_correctly_detected
        break;
    end
    
    current_window_start_idx = current_window_start_idx + round(0.5 * samples_per_ms);
end


% Part 4: XGBOOST - Sliding Window Analysis
fprintf('\n3b. Performing continuous analysis for XGBoost (Window: %.1f ms)...\n', window_duration_ms_xgboost);
current_window_start_idx = 1;
xgboost_correctly_detected = false;

while current_window_start_idx <= length(sim_result.Time) - window_size_samples_xgboost
    start_time_extraction = tic;
    
    start_idx = current_window_start_idx;
    end_idx = start_idx + window_size_samples_xgboost - 1;
    analysis_time = sim_result.Time(end_idx);

    feature_vector = extract_features_from_window(...
        sim_result, ...
        measurement_vars_to_analyze, ...
        start_idx, ...
        end_idx, ...
        decomposition_level, ...
        mother_wavelet ...
    );
    
    feature_extraction_times_xgb(end+1) = toc(start_time_extraction) * 1000;
    
    feature_vector_str = strjoin(cellstr(num2str(feature_vector(:))), ',');
    true_label_str = fixed_fault_type;

    [~, cmdout_predict] = system(sprintf('python "%s" "%s" "%s"', python_script_name, feature_vector_str, true_label_str));
    
    response_predict = jsondecode(cmdout_predict);
    
    if strcmp(response_predict.status, 'error')
        fprintf('Python Prediction Error: %s\n', response_predict.message);
        break;
    end
    
    times_since_inception_xgb(end+1) = analysis_time - fixed_inception_time;
    xgboost_predictions{end+1} = response_predict.xgboost_pred;
    xgboost_is_correct(end+1) = response_predict.is_xgboost_correct;
    prediction_times_with_load_xgb(end+1) = response_predict.load_time_ms + response_predict.prediction_time_ms;
    prediction_times_only_xgb(end+1) = response_predict.prediction_time_ms;

    idx_inception_sample = find(sim_result.Time >= fixed_inception_time, 1, 'first');
    is_after_inception = (current_window_start_idx >= idx_inception_sample);
    
    if ~is_after_inception
        if ~strcmp(response_predict.xgboost_pred, 'NoFault')
            xgboost_false_positives = xgboost_false_positives + 1;
        end
    else
        if ~xgboost_correctly_detected && xgboost_is_correct(end)
            detection_time_xgboost = times_since_inception_xgb(end);
            xgboost_correctly_detected = true;
        end
    end

    if xgboost_correctly_detected
        break;
    end
    
    current_window_start_idx = current_window_start_idx + round(0.5 * samples_per_ms);
end


% Part 5: Report Key Performance Metrics
fprintf('\n4. Key Performance Metrics\n');
fprintf('----------------------------------\n');
fprintf('Fault Type Tested: %s\n', fixed_fault_type);
fprintf('Fault Inception Time: %.3f s\n', fixed_inception_time);
fprintf('----------------------------------\n');
fprintf('CatBoost (Window: %.1f ms)\n', window_duration_ms_catboost);
fprintf('  False Positives: %d\n', catboost_false_positives);
if ~isnan(detection_time_catboost)
    fprintf('  Fault Detection Time (Latency): %.6f s (%.2f ms)\n', detection_time_catboost, detection_time_catboost * 1000);
else
    fprintf('  Did not detect the fault within the analysis window.\n');
end
fprintf('  Average Feature Extraction Time: %.4f ms\n', mean(feature_extraction_times_cat));
fprintf('  Average Model Loading + Prediction Time: %.4f ms\n', mean(prediction_times_with_load_cat));
fprintf('  Average Pure Model Prediction Time: %.4f ms\n', mean(prediction_times_only_cat));
fprintf('----------------------------------\n');
fprintf('XGBoost (Window: %.1f ms)\n', window_duration_ms_xgboost);
fprintf('  False Positives: %d\n', xgboost_false_positives);
if ~isnan(detection_time_xgboost)
    fprintf('  Fault Detection Time (Latency): %.6f s (%.2f ms)\n', detection_time_xgboost, detection_time_xgboost * 1000);
else
    fprintf('  Did not detect the fault within the analysis window.\n');
end
fprintf('  Average Feature Extraction Time: %.4f ms\n', mean(feature_extraction_times_xgb));
fprintf('  Average Model Loading + Prediction Time: %.4f ms\n', mean(prediction_times_with_load_xgb));
fprintf('  Average Pure Model Prediction Time: %.4f ms\n', mean(prediction_times_only_xgb));
fprintf('----------------------------------\n');

% Part 6: Generate Visual Outputs
fprintf('5. Generating plots for report...\n');

all_labels_from_meta = {'NoFault', 'Pole1_Pos_Fault', 'Pole1_Neg_Fault', 'Pole2_Pos_Fault', 'Pole2_Neg_Fault', 'Pole3_Pos_Fault', 'Pole3_Neg_Fault', 'Pole_to_Pole_Fault'};
all_labels = unique(all_labels_from_meta);
all_labels = sort(all_labels);
label_map = containers.Map(all_labels, 1:length(all_labels));
true_label_long_form = '';
switch fixed_fault_type
    case 'f_1p'
        true_label_long_form = 'Pole1_Pos_Fault';
    case 'f_1n'
        true_label_long_form = 'Pole1_Neg_Fault';
    case 'f_2p'
        true_label_long_form = 'Pole2_Pos_Fault';
    case 'f_2n'
        true_label_long_form = 'Pole2_Neg_Fault';
    case 'f_3p'
        true_label_long_form = 'Pole3_Pos_Fault';
    case 'f_3n'
        true_label_long_form = 'Pole3_Neg_Fault';
    case 'f_pp'
        true_label_long_form = 'Pole_to_Pole_Fault';
    otherwise
        true_label_long_form = 'NoFault';
end
true_label_mapped = label_map(true_label_long_form);


figure;
subplot(2,1,1);
plot(times_since_inception_cat * 1000, cell2mat(arrayfun(@(x) label_map(x{1}), catboost_predictions, 'UniformOutput', false)), ...
    '-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
xlabel('Time since inception (ms)');
ylabel('Predicted Label');
yticks(1:length(all_labels));
yticklabels(all_labels);
title(sprintf('CatBoost Predicted Label Over Time (Window: %.1f ms)', window_duration_ms_catboost));
grid on;
xline(0, 'r--', 'LineWidth', 2, 'Label', 'Fault Inception');
yline(true_label_mapped, 'k:', 'LineWidth', 1.5, 'Label', 'True Label');

subplot(2,1,2);
plot(times_since_inception_xgb * 1000, cell2mat(arrayfun(@(x) label_map(x{1}), xgboost_predictions, 'UniformOutput', false)), ...
    '-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
xlabel('Time since inception (ms)');
ylabel('Predicted Label');
yticks(1:length(all_labels));
yticklabels(all_labels);
title(sprintf('XGBoost Predicted Label Over Time (Window: %.1f ms)', window_duration_ms_xgboost));
grid on;
xline(0, 'r--', 'LineWidth', 2, 'Label', 'Fault Inception');
yline(true_label_mapped, 'k:', 'LineWidth', 1.5, 'Label', 'True Label');

sgtitle('Real-time Fault Detection Performance');
saveas(gcf, 'FaultDetectionPerformance.png');


figure;
subplot(1,2,1);
bar_data_cat = [mean(feature_extraction_times_cat), mean(prediction_times_only_cat)];
bar_labels = {'Feature Extraction', 'Model Prediction'};
bar(bar_data_cat);
set(gca, 'XTickLabel', bar_labels);
ylabel('Time (ms)');
title(sprintf('CatBoost Computational Burden (Window: %.1f ms)', window_duration_ms_catboost));
grid on;

subplot(1,2,2);
bar_data_xgb = [mean(feature_extraction_times_xgb), mean(prediction_times_only_xgb)];
bar_labels = {'Feature Extraction', 'Model Prediction'};
bar(bar_data_xgb);
set(gca, 'XTickLabel', bar_labels);
ylabel('Time (ms)');
title(sprintf('XGBoost Computational Burden (Window: %.1f ms)', window_duration_ms_xgboost));
grid on;
sgtitle('Computational Burden per Step (excluding load time)');

saveas(gcf, 'ComputationalBurden.png');

fprintf('Plots saved as FaultDetectionPerformance.png and ComputationalBurden.png.\n');

% --- HELPER FUNCTIONS ---
function result_struct = run_single_simulation(model_name_local, sim_duration_local, ...
        fault_type_local, fault_location_local, fault_inception_time_local_value, ...
        operating_condition_local, noise_level_percentage_local, sim_id_local)
    result_struct = struct();
    result_struct.SimID = sim_id_local;
    result_struct.Error = [];
    result_struct.Time = [];
    result_struct.FaultType = fault_type_local;
    result_struct.FaultLocation = fault_location_local;
    result_struct.FaultInceptionTime = fault_inception_time_local_value;
    result_struct.OperatingCondition = operating_condition_local;
    
    measurement_vars = {
        'Vdc_T1_Pos', 'Vdc_T1_Neg', 'Idc_T1_Pos', 'Idc_T1_Neg', ...
        'Vdc_T2_Pos', 'Vdc_T2_Neg', 'Idc_T2_Pos', 'Idc_T2_Neg', ...
        'Vdc_T3_Pos', 'Vdc_T3_Neg', 'Idc_T3_Pos', 'Idc_T3_Neg', ...
        'Vac_C1_A', 'Vac_C1_B', 'Vac_C1_C', 'Iac_C1_A', 'Iac_C1_B', 'Iac_C1_C', ...
        'Vac_C2_A', 'Vac_C2_B', 'Vac_C2_C', 'Iac_C2_A', 'Iac_C2_B', 'Iac_C2_C'
    };
    
    for m_var_idx = 1:length(measurement_vars)
        result_struct.(measurement_vars{m_var_idx}) = [];
    end

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
                assignin('base', 'fault_f1p_control', off_signal); 
                assignin('base', 'fault_f1n_control', off_signal);
                assignin('base', 'fault_f2p_control', off_signal); 
                assignin('base', 'fault_f2n_control', off_signal);
                assignin('base', 'fault_f3p_control', off_signal); 
                assignin('base', 'fault_f3n_control', off_signal);
                assignin('base', 'fault_pp_control', off_signal);
        end
        assignin('base', 'FAULT_INCEPTION_TIME', fault_inception_time_local_value);
    end

    set_param(model_name_local, 'StopTime', num2str(sim_duration_local));

    try
        sim_output_workspace = sim(model_name_local, 'ReturnWorkspaceOutputs', 'on');
    
        if isprop(sim_output_workspace, 'sim_time')
            result_struct.Time = sim_output_workspace.sim_time;
        elseif isprop(sim_output_workspace, 'logsout') && ~isempty(sim_output_workspace.logsout)
            if sim_output_workspace.logsout.find('Name', 'sim_time')
                result_struct.Time = sim_output_workspace.logsout.get('sim_time').Values.Data;
            end
        end

        if isempty(result_struct.Time)
            error('Could not obtain simulation time for Sim %d.', sim_id_local);
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

    catch ME
        result_struct.Error = ME.message;
        
        for m_var_idx = 1:length(measurement_vars)
            result_struct.(measurement_vars{m_var_idx}) = [];
        end
    end
end

function feature_vector = extract_features_from_window(sim_result, measurement_vars_to_analyze, start_idx, end_idx, decomposition_level, mother_wavelet)
    feature_vector = zeros(1, length(measurement_vars_to_analyze) * (decomposition_level + 2));
    current_feature_output_idx = 1;

    for j = 1:length(measurement_vars_to_analyze)
        signal_name = measurement_vars_to_analyze{j};
        analysis_window = sim_result.(signal_name)(start_idx:end_idx);

        if isrow(analysis_window)
            analysis_window = analysis_window';
        end
        if isempty(analysis_window) || all(analysis_window == 0) || any(isnan(analysis_window)) || any(isinf(analysis_window))
            current_signal_features = NaN(1, decomposition_level + 2);
        else
            [C, L] = wavedec(analysis_window, decomposition_level, mother_wavelet);
            energies_for_entropy = zeros(1, decomposition_level + 1);
            current_feature_idx = 1;
            current_signal_features = zeros(1, decomposition_level + 2);

            for level = 1:decomposition_level
                cD = detcoef(C, L, level);
                current_signal_features(current_feature_idx) = rms(cD);
                current_feature_idx = current_feature_idx + 1;
                energies_for_entropy(level) = sum(cD.^2);
            end

            cA_last_level_val = appcoef(C, L, mother_wavelet, decomposition_level);
            current_signal_features(current_feature_idx) = rms(cA_last_level_val);
            current_feature_idx = current_feature_idx + 1;
            energies_for_entropy(decomposition_level + 1) = sum(cA_last_level_val.^2);

            total_energy_for_entropy = sum(energies_for_entropy);
            if total_energy_for_entropy == 0
                entropy_val = 0;
            else
                p_energies = energies_for_entropy / total_energy_for_entropy;
                p_energies_nonzero = p_energies(p_energies > 0);
                if isempty(p_energies_nonzero)
                    entropy_val = 0;
                else
                    entropy_val = -sum(p_energies_nonzero .* log2(p_energies_nonzero), 'omitnan');
                end
            end
            current_signal_features(current_feature_idx) = entropy_val;
        end
        feature_vector(current_feature_output_idx : current_feature_output_idx + (decomposition_level + 2) - 1) = current_signal_features;
        current_feature_output_idx = current_feature_output_idx + (decomposition_level + 2);
    end
end