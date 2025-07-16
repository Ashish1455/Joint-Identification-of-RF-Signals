prmLTEPDSCH.Nrb = 50;      % Number of resource blocks (10 MHz)
prmLTEPDSCH.NcellID = 1;   % Cell ID
numSymbols = 14;           % OFDM symbols per subframe
fftSize = 1024;            % FFT size for 50 RBs
transportBlkSize = 1000;   % Transport block size (bits)
outlen = 2000;             % Output length after rate matching
rv = 0;                    % Redundancy version
numSignals = 100;          % Number of signals to generate

% Initialize cell array for signals
lteSignals = cell(numSignals, 1);

% Prepare cell array for CSV export
csvData = cell(numSignals+1, 4); % +1 for header row
csvData(1,:) = {'Signal', 'Input', 'ChannelCoding', 'Modulation'};

for idx = 1:numSignals
    % 1. Generate random transport block
    a = randi([0 1], transportBlkSize, 1);
    
    % 2. Channel encoding
    codedBits = lte_channel_coding_chain(a, outlen, rv);
    
    % 3. QPSK modulation
    modData = QPSK_Modulation(codedBits);
    
    % 4. Map to resource grid (frequency domain)
    in = zeros(fftSize, numSymbols);
    numData = min(length(modData), fftSize * numSymbols);
    in(1:numData) = modData(1:numData);
    
    % 5. OFDM modulation (with pilot placement)
    ofdmSignal = OFDMTx(in, prmLTEPDSCH);
    
    % 6. Store the generated signal
    lteSignals{idx} = ofdmSignal;
    
    % 7. Store data for CSV - Fixed serialization
    % Convert complex signal to string format (real+imag parts)
    signalStr = sprintf('%.6f%+.6fi ', real(ofdmSignal), imag(ofdmSignal));
    signalStr = signalStr(1:end-1); % Remove trailing space
    
    % Convert input bits to string
    inputStr = sprintf('%d', a');
    
    csvData{idx+1, 1} = signalStr;              % Serialized signal
    csvData{idx+1, 2} = inputStr;               % Serialized input bits
    csvData{idx+1, 3} = 'LTE Turbo Coding';     % Channel coding method
    csvData{idx+1, 4} = 'QPSK';                 % Modulation scheme
end

% Write to CSV file with proper escaping
fid = fopen('lteSignals.csv', 'w');
for i = 1:size(csvData,1)
    % Escape quotes in data by doubling them
    col1 = strrep(csvData{i,1}, '"', '""');
    col2 = strrep(csvData{i,2}, '"', '""');
    col3 = strrep(csvData{i,3}, '"', '""');
    col4 = strrep(csvData{i,4}, '"', '""');
    
    fprintf(fid, '"%s","%s","%s","%s"\n', col1, col2, col3, col4);
end
fclose(fid);

fprintf('CSV file created successfully with %d signals\n', numSignals);
