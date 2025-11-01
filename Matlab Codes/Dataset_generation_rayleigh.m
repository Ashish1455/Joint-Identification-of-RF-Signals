% MATLAB Code for Creating Communication Signal Dataset - RAYLEIGH FADING VERSION
% 12 classes: (4 modulations × 3 encoders) - NO INTERLEAVERS
% Ensures all classes generate signals with required length using Rayleigh fading
clc; clear; close all;

%% Check for required toolboxes
if ~license('test', 'communication_toolbox')
    error('Communications System Toolbox is required for this code');
end

%% Dataset Parameters
samples_per_snr = 3000;
signal_length = 1024;
min_inputs_size = 3*signal_length + 20;
SNR = -20:5:25;

% Define modulation schemes
modulators = {'bpsk', '8psk', '16qam', '64qam'};

% Define encoding schemes
encoders = {'turbo_31', 'conv_133_171', 'polar_128_1_2'};

% Rayleigh Channel Parameters
max_doppler_shift = 50; % Hz - typical for mobile communications
sample_rate = 10000; % Hz - adjust based on your application

num_classes = length(modulators) * length(encoders);
samples_per_class_per_snr = samples_per_snr / num_classes;

% Build class combinations
combinations = {};
class_idx = 1;
for m = 1:length(modulators)
    for e = 1:length(encoders)
        combinations{class_idx} = {modulators{m}, encoders{e}};
        class_idx = class_idx + 1;
    end
end

fprintf('Dataset Configuration (RAYLEIGH FADING):\n');
fprintf('- Total classes: %d (4 mod × 3 enc)\n', num_classes);
fprintf('- Signal length: %d symbols\n', signal_length);
fprintf('- SNR range: %d to %d dB (%d levels)\n', min(SNR), max(SNR), length(SNR));
fprintf('- Samples per SNR: %d\n', samples_per_snr);
fprintf('- Samples per class per SNR: ~%d\n', round(samples_per_class_per_snr));
fprintf('- Max Doppler shift: %d Hz\n', max_doppler_shift);
fprintf('- Sample rate: %d Hz\n', sample_rate);

%% Display class combinations
fprintf('\nClass combinations:\n');
for i = 1:num_classes
    fprintf('Class %2d: %s + %s\n', i-1, ...
        combinations{i}{1}, combinations{i}{2});
end

%% Create Rayleigh Channel Object
rayleigh_chan = comm.RayleighChannel( ...
    'SampleRate', sample_rate, ...
    'MaximumDopplerShift', max_doppler_shift, ...
    'PathDelays', 0, ...  % Single path (flat fading)
    'AveragePathGains', 0, ... % 0 dB gain
    'NormalizePathGains', true, ...
    'RandomStream', 'mt19937ar with seed', ...
    'Seed', 12345, ...
    'FadingTechnique', 'Sum of sinusoids');

%% Generate and save signals for each SNR level separately
for snr_idx = 1:length(SNR)
    snr_db = SNR(snr_idx);
    fprintf('\n=== Processing SNR: %d dB ===\n', snr_db);
    
    signals_real = zeros(samples_per_snr, signal_length);
    signals_imag = zeros(samples_per_snr, signal_length);
    labels_class = zeros(samples_per_snr, 1);
    
    % Reset channel for new SNR
    release(rayleigh_chan);
    
    signal_idx = 1;
    
    for class_num = 1:num_classes
        modulator_type = combinations{class_num}{1};
        encoder_type = combinations{class_num}{2};
        
        fprintf('Generating class %d/%d: %s + %s (SNR: %d dB, Rayleigh fading)\n', ...
            class_num-1, num_classes-1, modulator_type, encoder_type, snr_db);
        
        % Balanced allocation
        if class_num <= mod(samples_per_snr, num_classes)
            class_samples = ceil(samples_per_class_per_snr);
        else
            class_samples = floor(samples_per_class_per_snr);
        end
        
        stored = 0;
        while stored < class_samples && signal_idx <= samples_per_snr
            % Calculate required input data size with safety margin
            input_bits = calculate_input_size_safe(encoder_type, modulator_type, signal_length, min_inputs_size);
            success = false;
            retry_count = 0;
            
            while ~success && retry_count < 15
                retry_count = retry_count + 1;
                
                % Generate fresh random data bits (no repetition)
                data_bits = randi([0, 1], input_bits, 1);
                
                % Encode
                encoded_bits = apply_encoding_no_padding(data_bits, encoder_type);
                
                % Modulate
                modulated_signal = apply_modulation_no_padding(encoded_bits, modulator_type);
                
                % Check if we have enough symbols
                if length(modulated_signal) >= signal_length
                    success = true;
                else
                    % Increase input size and retry
                    input_bits = ceil(input_bits * 1.3);
                    fprintf('Retrying class %d with %d input bits (attempt %d)\n', ...
                        class_num-1, input_bits, retry_count);
                end
            end
            
            if ~success
                error('Failed to generate sufficient length for class %d after %d attempts', ...
                    class_num-1, retry_count);
            end
            
            % Trim to exact length (no padding, just truncate)
            modulated_signal = modulated_signal(1:signal_length);
            
            % Apply Rayleigh fading channel
            faded_signal = rayleigh_chan(modulated_signal);
            
            % Add AWGN noise to the faded signal
            faded_signal = awgn(faded_signal, snr_db, 'measured');
            
            % Store signals and labels
            signals_real(signal_idx, :) = real(faded_signal)';
            signals_imag(signal_idx, :) = imag(faded_signal)';
            labels_class(signal_idx) = class_num - 1;
            
            signal_idx = signal_idx + 1;
            stored = stored + 1;
        end
        
        if signal_idx > samples_per_snr
            break;
        end
    end
    
    % Save dataset
    filename = sprintf('dataset_Rayleigh_SNR_%d_dB.mat', snr_db);
    fprintf('Saving dataset to %s...\n', filename);
    save(filename, 'signals_real', 'signals_imag', 'labels_class', '-v7.3');
    fprintf('SNR %d dB dataset with Rayleigh fading saved! (%d signals)\n', snr_db, samples_per_snr);
end

%% Create summary file
total_signals = samples_per_snr * length(SNR);
summary_info = struct();
summary_info.total_signals = total_signals;
summary_info.samples_per_snr = samples_per_snr;
summary_info.snr_levels = SNR;
summary_info.num_classes = num_classes;
summary_info.signal_length = signal_length;
summary_info.combinations = combinations;
summary_info.channel_type = 'Rayleigh Fading';
summary_info.max_doppler_shift = max_doppler_shift;
summary_info.sample_rate = sample_rate;
summary_info.dataset_source = 'RAYLEIGH FADING + Guaranteed Length';
summary_info.files_created = {};

for i = 1:length(SNR)
    filename = sprintf('dataset_Rayleigh_SNR_%d_dB.mat', SNR(i));
    summary_info.files_created{i} = filename;
end

save('dataset_rayleigh_summary.mat', 'summary_info', '-v7.3');

fprintf('\n=== Rayleigh Fading Dataset Generation Complete! ===\n');
fprintf('Total signals generated: %d\n', total_signals);
fprintf('Classes: %d (0-%d)\n', num_classes, num_classes-1);
fprintf('Channel: Rayleigh fading with %d Hz max Doppler shift\n', max_doppler_shift);

%% Helper Functions
function input_bits = calculate_input_size_safe(encoder_type, modulator_type, target_length, min_inputs_size)
    % Get bits per symbol
    switch modulator_type
        case 'bpsk', bps = 1;
        case '8psk', bps = 3;
        case '16qam', bps = 4;
        case '64qam', bps = 6;
        otherwise, bps = 1;
    end
    
    encoded_bits_needed = target_length * bps;
    
    % Get approximate code rate
    switch encoder_type
        case 'conv_133_171', code_rate = 0.5;
        case 'turbo_31', code_rate = 0.33;
        case 'polar_128_1_2', code_rate = 0.5;
        otherwise, code_rate = 1.0;
    end
    
    % Calculate with large safety margin to avoid repetition/padding
    input_bits = ceil(encoded_bits_needed / code_rate * 1.5); % 50% safety margin
    input_bits = max(min_inputs_size, input_bits); % Minimum guarantee
end

function encoded_bits = apply_encoding_no_padding(data_bits, encoder_type)
    switch encoder_type
        case 'conv_133_171'
            trellis = poly2trellis(7, [171 133]);
            encoded_bits = convenc(data_bits, trellis);
            
        case 'turbo_31'
            feedback = 31; 
            generators = [15 13];
            trellis = poly2trellis(5, generators, feedback);
            u = data_bits(:); 
            K = length(u);
            
            e1 = convenc(u, trellis); 
            e1 = reshape(e1, [], length(generators));
            parity1 = e1(:, 2);
            
            rng(42); 
            idx = randperm(K).'; 
            u_int = u(idx);
            
            e2 = convenc(u_int, trellis); 
            e2 = reshape(e2, [], length(generators));
            parity2 = e2(:, 2);
            
            encoded_bits = reshape([u parity1 parity2].', [], 1);
            
        case 'polar_128_1_2'
            % Polar Code (128,64) - NO REPETITION OR PADDING
            N = 128; K = 64;
            encoded_bits = [];
            data_remaining = data_bits;
            
            while ~isempty(data_remaining) && length(encoded_bits) < 2 * length(data_bits)
                if length(data_remaining) >= K
                    block_data = data_remaining(1:K);
                    data_remaining(1:K) = [];
                else
                    % Use only available bits, no zero padding
                    block_data = [data_remaining; randi([0,1], K - length(data_remaining), 1)];
                    data_remaining = [];
                end
                
                % Map to polar codeword
                u = zeros(N, 1);
                u(1:K) = block_data;
                
                % Polar transform
                x = u;
                n = log2(N);
                for stage = 1:n
                    step = 2^stage;
                    half = step / 2;
                    for i = 1:step:N
                        idx1 = i:(i+half-1);
                        idx2 = (i+half):(i+step-1);
                        x(idx1) = mod(x(idx1) + x(idx2), 2);
                    end
                end
                encoded_bits = [encoded_bits; x(:)];
            end
            
        otherwise
            encoded_bits = data_bits;
    end
end

function modulated_signal = apply_modulation_no_padding(bits, modulator_type)
    switch modulator_type
        case 'bpsk'
            bps = 1;
            num_complete_symbols = floor(length(bits) / bps);
            if num_complete_symbols > 0
                bits_to_use = bits(1:num_complete_symbols);
                modulated_signal = pskmod(bits_to_use, 2);
            else
                modulated_signal = complex(zeros(0, 1));
            end
            
        case '8psk'
            bps = 3;
            num_complete_symbols = floor(length(bits) / bps);
            if num_complete_symbols > 0
                bits_to_use = bits(1:num_complete_symbols * bps);
                s = bi2de(reshape(bits_to_use, bps, []).', 'left-msb');
                modulated_signal = pskmod(s, 8, 0, 'gray');
            else
                modulated_signal = complex(zeros(0, 1));
            end
            
        case '16qam'
            bps = 4;
            num_complete_symbols = floor(length(bits) / bps);
            if num_complete_symbols > 0
                bits_to_use = bits(1:num_complete_symbols * bps);
                s = bi2de(reshape(bits_to_use, bps, []).', 'left-msb');
                modulated_signal = qammod(s, 16, 'gray');
            else
                modulated_signal = complex(zeros(0, 1));
            end
            
        case '64qam'
            bps = 6;
            num_complete_symbols = floor(length(bits) / bps);
            if num_complete_symbols > 0
                bits_to_use = bits(1:num_complete_symbols * bps);
                s = bi2de(reshape(bits_to_use, bps, []).', 'left-msb');
                modulated_signal = qammod(s, 64, 'gray');
            else
                modulated_signal = complex(zeros(0, 1));
            end
            
        otherwise
            modulated_signal = complex(bits);
    end
    
    modulated_signal = modulated_signal(:);
end
