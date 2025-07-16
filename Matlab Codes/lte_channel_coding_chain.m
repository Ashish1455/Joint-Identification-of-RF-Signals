function f = lte_channel_coding_chain(a, outlen, rv)
    % a: Input transport block (column vector of bits)
    % outlen: Output length after rate matching (scalar)
    % rv: Redundancy version for rate matching (0,1,2,3)
    % f: Output codeword after all processing

    % 1. Transport block CRC attachment (24A)
    b = lteCRCEncode(a, '24A'); % [1][23]

    % 2. Code block segmentation and code block CRC attachment (24B)
    cbs = lteCodeBlockSegment(b); % [3][9]

    % 3. Channel coding (Turbo encoding)
    for i = 1:length(cbs)
        cbs{i} = lteTurboEncode(cbs{i}); % [33][29]
    end

    % 4. Rate matching (Turbo)
    for i = 1:length(cbs)
        % Each code block may have a different output length; here, we split equally
        outlen_i = floor(outlen / length(cbs));
        d{i} = lteRateMatchTurbo(cbs{i}, outlen_i, rv); % [11]
    end

    % 5. Code block concatenation
    f = vertcat(d{:}); % [7]
end
