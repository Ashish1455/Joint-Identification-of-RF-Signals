% MATLAB Code for Creating Communication Signal Dataset - DETERMINISTIC VERSION WITH RAYLEIGH FADING
% All encoders now produce identical outputs for identical inputs
% Uses Rayleigh fading channel instead of AWGN

%% ========================= ALL DEFINITIONS AND PARAMETERS =========================

% Clear environment and suppress warnings
clc; clear; close all;
warning('off', 'all');

% Toolbox check
required_toolbox = 'communication_toolbox';

% Main dataset parameters
SAMPLES_PER_SNR = 3000;              % Total samples per SNR level
SIGNAL_LENGTH = 256;                 % Length of each signal in symbols
MIN_INPUT_SIZE = 3 * SIGNAL_LENGTH + 20;  % Minimum input size for encoding
SNR_RANGE = -20:5:25;                 % SNR levels in dB
SAMPLES_PER_CLASS = 250;               % Fixed samples per class per SNR (4000/12 ≈ 333)

% Signal processing parameters 
SAFETY_MARGIN = 1.5;                  % 50% safety margin for input calculation
MAX_RETRY_ATTEMPTS = 15;              % Maximum retry attempts for signal generation
INPUT_SIZE_MULTIPLIER = 1.3;          % Multiplier for input size increase on retry

% Deterministic behavior parameters
DETERMINISTIC_SEED = 42;              % Base seed for all random operations

% Rayleigh fading parameters
SAMPLE_RATE = 1e6;                    % Sample rate in Hz
MAX_DOPPLER_SHIFT = 100;              % Maximum Doppler shift in Hz

% Encoder parameters
CONV_CONSTRAINT_LENGTH = 7;
CONV_GENERATOR_POLY = [171, 133];
TURBO_CONSTRAINT_LENGTH = 5;
TURBO_FEEDBACK_POLY = 31;
TURBO_GENERATOR_POLY = [15, 13];
POLAR_N = 128;
POLAR_K = 64;

% Define modulation schemes with their parameters
MODULATION_SCHEMES = {'bpsk', '8psk', '16qam', '64qam'};
MODULATION_BPS = [1, 3, 4, 6];       % Bits per symbol for each modulation

% Define encoding schemes with their code rates (LDPC removed)
ENCODING_SCHEMES = {'turbo_31', 'conv_133_171', 'polar_128_1_2'};
CODE_RATES = [0.33, 0.5, 0.5];  % Code rates for each encoder

% Calculate derived parameters
NUM_MODULATIONS = length(MODULATION_SCHEMES);
NUM_ENCODERS = length(ENCODING_SCHEMES);
NUM_CLASSES = NUM_MODULATIONS * NUM_ENCODERS; % now 12 classes (4 mod × 3 enc)
SAMPLES_PER_CLASS_PER_SNR = SAMPLES_PER_SNR / NUM_CLASSES;
NUM_SNR_LEVELS = length(SNR_RANGE);
TOTAL_SIGNALS = SAMPLES_PER_SNR * NUM_SNR_LEVELS;

% File naming
MAT_VERSION = '-v7.3';
DATASET_PREFIX = 'dataset_rayleigh_SNR_noldpc';
SUMMARY_FILENAME = 'dataset_rayleigh_summary_noldpc.mat';

% Messages
ERROR_TOOLBOX = 'Communications System Toolbox is required for this code';
MSG_DATASET_CONFIG = 'Deterministic Dataset Configuration with Rayleigh Fading:';

%% ========================= MAIN EXECUTION =========================

%% Check for required toolboxes
if ~license('test', required_toolbox)
    error(ERROR_TOOLBOX);
end

%% Display dataset configuration
fprintf([MSG_DATASET_CONFIG '\n']);
fprintf('- DETERMINISTIC ENCODERS: Same input = Same output\n');
fprintf('- CHANNEL MODEL: Rayleigh Fading + AWGN\n');
fprintf('- Total classes: %d (%d mod × %d enc)\n', NUM_CLASSES, NUM_MODULATIONS, NUM_ENCODERS);
fprintf('- Signal length: %d symbols\n', SIGNAL_LENGTH);
fprintf('- SNR range: %d to %d dB (%d levels)\n', min(SNR_RANGE), max(SNR_RANGE), NUM_SNR_LEVELS);
fprintf('- Samples per SNR: %d\n', SAMPLES_PER_SNR);
fprintf('- Max Doppler shift: %d Hz\n', MAX_DOPPLER_SHIFT);

%% Build class combinations
combinations = {};
class_idx = 1;
for m = 1:NUM_MODULATIONS
    for e = 1:NUM_ENCODERS
        combinations{class_idx} = {MODULATION_SCHEMES{m}, ENCODING_SCHEMES{e}};
        class_idx = class_idx + 1;
    end
end

%% Display class combinations
fprintf('\nClass combinations:\n');
for i = 1:NUM_CLASSES
    fprintf('Class %2d: %s + %s\n', i-1, combinations{i}{1}, combinations{i}{2});
end

%% Generate and save signals for each SNR level separately
for snr_idx = 1:NUM_SNR_LEVELS
    snr_db = SNR_RANGE(snr_idx);
    fprintf('\n=== Processing SNR: %d dB ===\n', snr_db);

    % Initialize signal storage arrays
    signals_real = zeros(SAMPLES_PER_SNR, SIGNAL_LENGTH);
    signals_imag = zeros(SAMPLES_PER_SNR, SIGNAL_LENGTH);
    labels_class = zeros(SAMPLES_PER_SNR, 1);
    signal_idx = 1;

    % Generate signals for each class
    for class_num = 1:NUM_CLASSES
        modulator_type = combinations{class_num}{1};
        encoder_type = combinations{class_num}{2};

        fprintf('Generating class %d/%d: %s + %s\n', class_num-1, NUM_CLASSES-1, modulator_type, encoder_type);

        stored = 0;
        while stored < SAMPLES_PER_CLASS && signal_idx <= SAMPLES_PER_SNR
            % Calculate required input data size
            input_bits = calculate_input_size_safe(encoder_type, modulator_type, SIGNAL_LENGTH, MIN_INPUT_SIZE, SAFETY_MARGIN);

            success = false;
            retry_count = 0;

            while ~success && retry_count < MAX_RETRY_ATTEMPTS
                retry_count = retry_count + 1;

                % Generate deterministic random data bits based on signal index
                % This ensures reproducibility while still having variety
                rng(DETERMINISTIC_SEED + signal_idx + class_num * 1000);
                data_bits = randi([0, 1], input_bits, 1);

                % DETERMINISTIC ENCODING - same input always produces same output
                encoded_bits = apply_encoding_deterministic(data_bits, encoder_type);

                % Modulate
                modulated_signal = apply_modulation_no_padding(encoded_bits, modulator_type);

                % Check if we have enough symbols
                if length(modulated_signal) >= SIGNAL_LENGTH
                    success = true;
                else
                    input_bits = ceil(input_bits * INPUT_SIZE_MULTIPLIER);
                end
            end

            if ~success
                error('Failed to generate sufficient length for class %d', class_num-1);
            end

            % Trim to exact length
            modulated_signal = modulated_signal(1:SIGNAL_LENGTH);

            % Apply Rayleigh fading channel with AWGN
            modulated_signal = apply_rayleigh_fading(modulated_signal, snr_db, signal_idx);

            % Store signals and labels
            signals_real(signal_idx, :) = real(modulated_signal)';
            signals_imag(signal_idx, :) = imag(modulated_signal)';
            labels_class(signal_idx) = class_num - 1;

            signal_idx = signal_idx + 1;
            stored = stored + 1;
        end
    end

    % Save dataset
    filename = sprintf([DATASET_PREFIX '%d_dB.mat'], snr_db);
    save(filename, 'signals_real', 'signals_imag', 'labels_class', MAT_VERSION);
    fprintf('Rayleigh fading dataset saved: %s\n', filename);
end

fprintf('\n=== Rayleigh Fading Dataset Generation Complete! ===\n');

%% ========================= DETERMINISTIC HELPER FUNCTIONS =========================

function input_bits = calculate_input_size_safe(encoder_type, modulator_type, target_length, min_inputs_size, safety_margin)
    modulation_schemes = {'bpsk', '8psk', '16qam', '64qam'};
    bits_per_symbol = [1, 3, 4, 6];

    mod_idx = find(strcmp(modulator_type, modulation_schemes));
    if isempty(mod_idx)
        bps = 1;
    else
        bps = bits_per_symbol(mod_idx);
    end

    encoded_bits_needed = target_length * bps;

    encoding_schemes = {'conv_133_171', 'turbo_31', 'polar_128_1_2'};
    code_rates = [0.5, 0.33, 0.5];

    enc_idx = find(strcmp(encoder_type, encoding_schemes));
    if isempty(enc_idx)
        code_rate = 1.0;
    else
        code_rate = code_rates(enc_idx);
    end

    input_bits = ceil(encoded_bits_needed / code_rate * safety_margin);
    input_bits = max(min_inputs_size, input_bits);
end

function faded_signal = apply_rayleigh_fading(signal, snr_db, seed_offset)
    % Apply Rayleigh fading channel with AWGN noise
    % Uses deterministic seed for reproducible fading
    
    DETERMINISTIC_SEED = 42;
    MAX_DOPPLER_SHIFT = 100;
    SAMPLE_RATE = 1e6;
    
    % Set deterministic seed for fading channel
    rng(DETERMINISTIC_SEED + seed_offset);
    
    % Create Rayleigh fading channel object
    rayleighchan = comm.RayleighChannel( ...
        'SampleRate', SAMPLE_RATE, ...
        'MaximumDopplerShift', MAX_DOPPLER_SHIFT, ...
        'RandomStream', 'mt19937ar with seed', ...
        'Seed', DETERMINISTIC_SEED + seed_offset);
    
    % Apply Rayleigh fading
    faded_signal = rayleighchan(signal);
    
    % Add AWGN noise
    faded_signal = awgn(faded_signal, snr_db, 'measured');
end

function encoded_bits = apply_encoding_deterministic(data_bits, encoder_type)
    % DETERMINISTIC ENCODER - Same input always produces same output
    CONV_CONSTRAINT_LENGTH = 7;
    CONV_GENERATOR_POLY = [171, 133];
    TURBO_CONSTRAINT_LENGTH = 5;
    TURBO_FEEDBACK_POLY = 31;
    TURBO_GENERATOR_POLY = [15, 13];
    POLAR_N = 128;
    POLAR_K = 64;
    DETERMINISTIC_SEED = 42;

    switch encoder_type
        case 'conv_133_171'
            trellis = poly2trellis(CONV_CONSTRAINT_LENGTH, CONV_GENERATOR_POLY);
            encoded_bits = convenc(data_bits, trellis);

        case 'turbo_31'
            feedback = TURBO_FEEDBACK_POLY;
            generators = TURBO_GENERATOR_POLY;
            trellis = poly2trellis(TURBO_CONSTRAINT_LENGTH, generators, feedback);
            u = data_bits(:);
            K = length(u);

            e1 = convenc(u, trellis);
            e1 = reshape(e1, [], length(generators));
            parity1 = e1(:, 2);

            % DETERMINISTIC: Seed based on input characteristics
            data_checksum = mod(sum(data_bits) + K, 1000);
            rng(DETERMINISTIC_SEED + data_checksum);
            idx = randperm(K).';
            u_int = u(idx);

            e2 = convenc(u_int, trellis);
            e2 = reshape(e2, [], length(generators));
            parity2 = e2(:, 2);

            encoded_bits = reshape([u parity1 parity2].', [], 1);

        case 'polar_128_1_2'
            N = POLAR_N;
            K = POLAR_K;
            encoded_bits = [];
            data_remaining = data_bits;

            while ~isempty(data_remaining) && length(encoded_bits) < 2 * length(data_bits)
                if length(data_remaining) >= K
                    block_data = data_remaining(1:K);
                    data_remaining(1:K) = [];
                else
                    % DETERMINISTIC padding
                    padding_length = K - length(data_remaining);
                    if length(data_remaining) > 0
                        repetitions = ceil(padding_length / length(data_remaining));
                        padding = repmat(data_remaining, repetitions, 1);
                        padding = padding(1:padding_length);
                    else
                        % Use only available bits, no zero padding
                        block_data = [data_remaining; randi([0,1], K - length(data_remaining), 1)];
                        data_remaining = [];
                    end
                    block_data = [data_remaining; padding];
                    data_remaining = [];
                end

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

        % LDPC encoder removed to simplify dataset (handled by other encoders)

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