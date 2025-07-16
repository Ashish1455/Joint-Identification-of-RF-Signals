function y = OFDMTx(in, prmLTEPDSCH)
% OFDM Transmitter for LTE with pilot (reference signal) placement
% in: Input data grid (frequency domain)
% prmLTEPDSCH: Structure with LTE parameters (e.g., Nrb, NcellID)

    % Determine FFT size and cyclic prefix based on number of resource blocks
    switch prmLTEPDSCH.Nrb
        case 25
            N = 512;
            cpLen = 36;
        case 50
            N = 1024;
            cpLen = 72;
        case 100
            N = 2048;
            cpLen = 144;
        otherwise
            % error('Unsupported Nrb value');
    end

    % IFFT processing to create OFDM signal
    X = ifft(in, N, 1);

    % Add cyclic prefix
    numSymbols = size(X, 2);
    y = [];
    for k = 1:numSymbols
        cp = X(end-cpLen+1:end, k);
        y = [y; cp; X(:, k)];
    end
end