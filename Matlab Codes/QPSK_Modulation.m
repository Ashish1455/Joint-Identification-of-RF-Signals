function symbols = QPSK_Modulation(bits)
% QPSK_Modulation - Maps input bits to QPSK symbols (Gray mapping)
% Input:
%   bits - Column vector of bits (0s and 1s), length must be even
% Output:
%   symbols - Column vector of QPSK symbols (complex values)

    % Input validation
    if mod(length(bits), 2) ~= 0
        error('Input bit vector length must be even.');
    end

    % Reshape bits into pairs
    bits = bits(:); % Ensure column vector
    bitPairs = reshape(bits, 2, []).';

    % Gray mapping for QPSK:
    % 00 -> 1 + 1j
    % 01 -> -1 + 1j
    % 11 -> -1 - 1j
    % 10 -> 1 - 1j
    mapping = [1+1j; -1+1j; -1-1j; 1-1j];

    % Convert bit pairs to decimal indices
    idx = bitPairs(:,1)*2 + bitPairs(:,2) + 1;

    % Map to QPSK symbols
    symbols = mapping(idx);

    % Normalize to unit average power
    symbols = symbols / sqrt(2);
end
